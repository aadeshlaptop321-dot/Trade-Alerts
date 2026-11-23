"""
Breakout alerts -> Telegram.   Runs on GitHub Actions (see .github/workflows/alerts.yml).

What it checks on 5-minute CLOSED candles (never on a half-formed candle):
  1. Previous-day high / low break           (all instruments)
  2. Opening-range (first 30 min) break      (Nifty, Bank Nifty, Sensex)
  3. Supertrend (10,3) flip                  (all instruments)
  4. Your own levels in levels.json          (all instruments)
  + a morning brief at ~9:10 AM IST

Rules: 0.1% buffer past a level; no alerts in the first 15 min of a session or after the
window ends; a trigger can fire again only after price has gone back inside and breaks
again, at least 30 min later; at most 6 alerts per instrument per day.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
HERE = Path(__file__).parent
BAR = pd.Timedelta(minutes=5)
T = dt.time

INSTRUMENTS = {
    "NIFTY 50":        dict(yf="^NSEI",   market="NSE"),
    "BANK NIFTY":      dict(yf="^NSEBANK", market="NSE"),
    "SENSEX":          dict(yf="^BSESN",  market="NSE"),
    "GOLD (COMEX)":    dict(yf="GC=F",    market="MCX"),
    "SILVER (COMEX)":  dict(yf="SI=F",    market="MCX"),
    "CRUDE OIL (WTI)": dict(yf="CL=F",    market="MCX"),
    "NATURAL GAS":     dict(yf="NG=F",    market="MCX"),
}
SESSION = {"NSE": (T(9, 15), T(15, 30)), "MCX": (T(9, 0), T(23, 30))}      # candle START inside this
WINDOW = {"NSE": (T(9, 30), T(15, 0)), "MCX": (T(9, 15), T(23, 30))}       # candle CLOSE inside this

DEFAULT_CFG = dict(
    buffer_pct=0.1,        # % beyond the level the candle must close
    min_gap_min=30,        # same trigger+direction: minimum minutes between alerts
    daily_cap=6,           # max alerts per instrument per day
    orb_minutes=30,        # opening-range length (NSE)
    catchup_min=20,        # re-check candles closed within this many minutes (covers a late run)
    drop_after_min=60,     # older than this -> don't alert (stale)
    morning_brief=True,
    instruments=list(INSTRUMENTS),
    triggers=dict(prev_day=True, opening_range=True, supertrend=True, custom=True),
)


# ----------------------------------------------------------------------------- data
def fetch(ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    import yfinance as yf
    last = None
    for _ in range(3):
        try:
            df = yf.Ticker(ticker).history(interval=interval, period=period, auto_adjust=False)
            if df is not None and not df.empty:
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
                idx = df.index
                if idx.tz is None:
                    idx = idx.tz_localize("UTC")
                df.index = idx.tz_convert(IST).tz_localize(None)       # naive IST wall clock
                return df
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"no data for {ticker}" + (f": {last}" if last else ""))


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    delta = c.diff()
    rs = _rma(delta.clip(lower=0), 14) / _rma(-delta.clip(upper=0), 14).replace(0, np.nan)
    d["RSI"] = (100 - 100 / (1 + rs)).fillna(100)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _rma(tr, 10)
    hl2 = (h + l) / 2
    ub, lb = (hl2 + 3 * atr).values, (hl2 - 3 * atr).values
    cl = c.values
    fu, fl = ub.copy(), lb.copy()
    dirn = np.ones(len(d), dtype=int)
    for i in range(1, len(d)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or cl[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or cl[i - 1] < fl[i - 1]) else fl[i - 1]
        if dirn[i - 1] == 1:
            dirn[i] = -1 if cl[i] < fl[i] else 1
        else:
            dirn[i] = 1 if cl[i] > fu[i] else -1
    d["ST_dir"] = dirn
    d["ST"] = np.where(dirn == 1, fl, fu)
    d["ATR"] = atr
    return d


# ----------------------------------------------------------------------------- levels
def session_bars(df: pd.DataFrame, market: str) -> pd.DataFrame:
    lo, hi = SESSION[market]
    t = df.index.time
    return df[[(x >= lo and x < hi) for x in t]]


def day_levels(df: pd.DataFrame, day: dt.date, market: str, orb_min: int) -> dict:
    sb = session_bars(df, market)
    out = {}
    prev = sb[sb.index.date < day]
    if not prev.empty:
        pd_day = prev[prev.index.date == prev.index[-1].date()]
        out.update(pdh=float(pd_day["High"].max()), pdl=float(pd_day["Low"].min()),
                   pdc=float(pd_day["Close"].iloc[-1]), pd_date=prev.index[-1].date())
    if market == "NSE":
        today = sb[sb.index.date == day]
        if not today.empty:
            start = dt.datetime.combine(day, SESSION[market][0])
            orng = today[today.index < start + pd.Timedelta(minutes=orb_min)]
            if len(orng) >= max(1, orb_min // 5 - 1):
                out.update(or_high=float(orng["High"].max()), or_low=float(orng["Low"].min()),
                           or_end=start + pd.Timedelta(minutes=orb_min))
    return out


# ----------------------------------------------------------------------------- alert text
def fmt(x: float) -> str:
    return f"{x:,.2f}" if abs(x) < 1000 else f"{x:,.0f}"


def plan_lines(up: bool, level: float, close: float, atr: float, rsi: float, stop_line: float | None = None) -> list[str]:
    buf = max(atr * 0.3, close * 0.0003)
    stop = stop_line if stop_line is not None else (level - buf if up else level + buf)
    risk = abs(close - stop)
    t1, t2 = (close + risk, close + 2 * risk) if up else (close - risk, close - 2 * risk)
    lines = [f"Stop {fmt(stop)}  |  T1 {fmt(t1)}  |  T2 {fmt(t2)}  (1R / 2R)"]
    side = "below" if up else "above"
    lines.append(f"Wrong if a 5-min candle closes back {side} {fmt(level)}")
    if up and rsi >= 75:
        lines.append(f"Careful: RSI {rsi:.0f} is overbought - don't chase")
    if (not up) and rsi <= 25:
        lines.append(f"Careful: RSI {rsi:.0f} is oversold - don't chase")
    return lines


def make_alert(name, up, title, level, bar, close_t, extra, stop_line=None):
    icon = "🟢" if up else "🔴"
    head = f"{icon} {name} - {title}"
    body = [f"Level {fmt(level)}  |  Close {fmt(float(bar['Close']))}  "
            f"(5m candle {close_t - BAR:%H:%M}-{close_t:%H:%M})"]
    body += plan_lines(up, level, float(bar["Close"]), float(bar["ATR"]), float(bar["RSI"]), stop_line)
    st = "up" if bar["ST_dir"] == 1 else "down"
    body.append(f"RSI {float(bar['RSI']):.0f}  |  Supertrend {st}" + (f"  |  {extra}" if extra else ""))
    return head + "\n" + "\n".join(body)


# ----------------------------------------------------------------------------- engine
def evaluate(name: str, df: pd.DataFrame, state: dict, now: dt.datetime, cfg: dict,
             custom_levels: list[dict] | None = None) -> list[str]:
    """Return alert texts for candles that closed recently; updates `state` (dedupe)."""
    info = INSTRUMENTS[name]
    market = info["market"]
    d = add_indicators(df)
    now_n = now.replace(tzinfo=None)
    d = d[d.index + BAR <= now_n]                       # closed candles only
    if len(d) < 3:
        return []
    sent = state.setdefault("sent", {})
    count = state.setdefault("count", {})
    gap = pd.Timedelta(minutes=cfg["min_gap_min"])
    w_lo, w_hi = WINDOW[market]
    trig = cfg["triggers"]
    buf = cfg["buffer_pct"] / 100.0
    out: list[str] = []
    lv_cache: dict = {}

    positions = [i for i in range(1, len(d)) if d.index[i] + BAR >= now_n - pd.Timedelta(minutes=cfg["catchup_min"])]
    for i in positions:
        bar, prev = d.iloc[i], d.iloc[i - 1]
        close_t = d.index[i] + BAR
        if not (w_lo <= close_t.time() <= w_hi):
            continue
        age = (now_n - close_t).total_seconds() / 60
        if age > cfg["drop_after_min"]:
            continue
        day = d.index[i].date()
        if day not in lv_cache:
            lv_cache[day] = day_levels(d, day, market, cfg["orb_minutes"])
        lv = lv_cache[day]
        cnt_key = f"{name}|{day}"
        c, pc = float(bar["Close"]), float(prev["Close"])

        events = []   # (key, up, title, level, extra, stop_line)
        if trig.get("prev_day") and "pdh" in lv:
            hi, lo = lv["pdh"] * (1 + buf), lv["pdl"] * (1 - buf)
            if c > hi and pc <= hi:
                events.append(("pdh|up", True, "broke above previous-day high", lv["pdh"], "", None))
            if c < lo and pc >= lo:
                events.append(("pdl|dn", False, "broke below previous-day low", lv["pdl"], "", None))
        if trig.get("opening_range") and "or_high" in lv and d.index[i] >= lv["or_end"]:
            hi, lo = lv["or_high"] * (1 + buf), lv["or_low"] * (1 - buf)
            rng = f"opening range {fmt(lv['or_low'])}-{fmt(lv['or_high'])}"
            if c > hi and pc <= hi:
                events.append(("orb|up", True, "broke above opening range high", lv["or_high"], rng, None))
            if c < lo and pc >= lo:
                events.append(("orb|dn", False, "broke below opening range low", lv["or_low"], rng, None))
        if trig.get("supertrend") and bar["ST_dir"] != prev["ST_dir"]:
            up = bar["ST_dir"] == 1
            events.append(("st|up" if up else "st|dn", up,
                           "Supertrend flipped " + ("UP" if up else "DOWN") + " (5m)",
                           float(bar["ST"]), "", float(bar["ST"])))
        if trig.get("custom"):
            for cl in custom_levels or []:
                if cl.get("instrument", "").upper() != name.upper():
                    continue
                L = float(cl["level"])
                dirn = str(cl.get("direction", "above")).lower()
                note = cl.get("note", "")
                if dirn == "above" and c > L and pc <= L:
                    events.append((f"c|{L}|up", True, f"crossed above your level {fmt(L)}", L, note, None))
                if dirn == "below" and c < L and pc >= L:
                    events.append((f"c|{L}|dn", False, f"crossed below your level {fmt(L)}", L, note, None))

        for key, up, title, level, extra, stop_line in events:
            k = f"{name}|{key}"
            last = sent.get(k)
            if last and close_t <= pd.Timestamp(last) + gap:
                continue                                  # same candle re-checked, or too soon
            if count.get(cnt_key, 0) >= cfg["daily_cap"]:
                continue
            text = make_alert(name, up, title, level, bar, close_t, extra, stop_line)
            if age > 8:
                text = f"(delayed {age:.0f} min)\n" + text
            out.append(text)
            sent[k] = close_t.isoformat()
            count[cnt_key] = count.get(cnt_key, 0) + 1
    return out


def morning_brief(frames: dict[str, pd.DataFrame], dailies: dict[str, pd.DataFrame], now: dt.datetime, cfg: dict) -> str:
    lines = [f"☀️ Morning brief - {now:%a %d %b}"]
    for name, df in frames.items():
        mk = INSTRUMENTS[name]["market"]
        lv = day_levels(df, now.date(), mk, cfg["orb_minutes"])
        if "pdh" not in lv:
            continue
        bias = ""
        dd = dailies.get(name)
        if dd is not None and len(dd) > 55:
            c = dd["Close"]
            e20, e50 = c.ewm(span=20, adjust=False).mean().iloc[-1], c.ewm(span=50, adjust=False).mean().iloc[-1]
            last = float(c.iloc[-1])
            bias = "Bullish" if last > e20 > e50 else "Bearish" if last < e20 < e50 else "Mixed"
            bias = f"  |  daily trend {bias}"
        lines.append(f"\n{name}\n  High {fmt(lv['pdh'])}  Low {fmt(lv['pdl'])}  Close {fmt(lv['pdc'])}{bias}")
    lines.append("\nAlerts fire on 5-min candle closes beyond the previous high/low, the opening range "
                 "(NSE, from 9:45), Supertrend flips and your own levels.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- io
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def telegram(text: str, dry: bool = False) -> bool:
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    print("----- ALERT -----\n" + text)
    if dry or not token or not chat:
        if not dry:
            print("(TELEGRAM_TOKEN / TELEGRAM_CHAT_ID not set - printed only)")
        return False
    for _ in range(2):
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              data={"chat_id": chat, "text": text}, timeout=20)
            if r.ok:
                return True
            print("Telegram error:", r.status_code, r.text[:200])
        except Exception as e:  # noqa: BLE001
            print("Telegram error:", e)
    return False


def prune(state: dict, today: dt.date) -> None:
    keep = str(today)
    state["count"] = {k: v for k, v in state.get("count", {}).items() if k.endswith(keep)}
    cutoff = (pd.Timestamp(today) - pd.Timedelta(days=3)).isoformat()
    state["sent"] = {k: v for k, v in state.get("sent", {}).items() if v >= cutoff}


def run(args) -> int:
    cfg = {**DEFAULT_CFG, **load_json(HERE / "config.json", {})}
    cfg["triggers"] = {**DEFAULT_CFG["triggers"], **cfg.get("triggers", {})}
    levels = load_json(HERE / "levels.json", [])
    levels = [x for x in levels if isinstance(x, dict) and "level" in x]
    state_path = Path(args.state)
    state = load_json(state_path, {})
    now = dt.datetime.now(IST) if not args.now else dt.datetime.fromisoformat(args.now).replace(tzinfo=IST)

    if args.test:
        telegram(f"✅ Alerts are connected ({now:%d %b %H:%M} IST). You will get breakout alerts here.", args.dry)
        return 0
    if now.weekday() >= 5:
        print("Weekend - nothing to do")
        return 0

    msgs: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    fails = state.setdefault("fail", {})
    stale = state.setdefault("stale", {})
    for name in cfg["instruments"]:
        if name not in INSTRUMENTS:
            continue
        try:
            df = fetch(INSTRUMENTS[name]["yf"])
            fails[name] = 0
        except Exception as e:  # noqa: BLE001
            fails[name] = fails.get(name, 0) + 1
            print(f"{name}: {e}")
            if fails[name] == 6 and stale.get(f"fail|{name}") != str(now.date()):
                stale[f"fail|{name}"] = str(now.date())
                msgs.append(f"⚠️ {name}: price data has failed for the last 6 checks. Alerts for it are paused.")
            continue
        frames[name] = df
        mk = INSTRUMENTS[name]["market"]
        lo, hi = WINDOW[mk]
        last_close = df.index[-1] + BAR
        lag = (now.replace(tzinfo=None) - last_close).total_seconds() / 60
        if lo <= now.time() <= hi and lag > 20 and stale.get(name) != str(now.date()):
            stale[name] = str(now.date())
            msgs.append(f"⚠️ {name}: latest price data is {lag:.0f} min old (Yahoo is delayed). "
                        "Alerts for it will be late today.")
        msgs += evaluate(name, df, state, now, cfg, levels)

    mb = state.get("morning")
    if cfg["morning_brief"] and mb != str(now.date()) and T(9, 5) <= now.time() <= T(9, 45) and frames:
        dailies = {}
        for n in frames:
            try:
                dailies[n] = fetch(INSTRUMENTS[n]["yf"], "1d", "1y")
            except Exception:  # noqa: BLE001
                pass
        msgs.append(morning_brief(frames, dailies, now, cfg))
        state["morning"] = str(now.date())

    for m in msgs:
        telegram(m, args.dry)
    prune(state, now.date())
    state_path.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    print(f"done: {len(msgs)} message(s)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(HERE / "state.json"))
    ap.add_argument("--test", action="store_true", help="send a test message and exit")
    ap.add_argument("--dry", action="store_true", help="print, do not send")
    ap.add_argument("--now", help="override clock, IST, e.g. 2026-11-24T10:06:00")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
