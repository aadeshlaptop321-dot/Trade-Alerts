import datetime as dt, sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import alerts as A

CFG = {**A.DEFAULT_CFG}
def day_bars(day, closes, start=dt.time(9, 15), spread=5.0):
    t0 = dt.datetime.combine(day, start)
    rows = []
    prev = closes[0]
    for i, c in enumerate(closes):
        o = prev
        rows.append((t0 + pd.Timedelta(minutes=5 * i), o, max(o, c) + spread / 5, min(o, c) - spread / 5, c, 0))
        prev = c
    return pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close", "Volume"]).set_index("t")

def frame(prev_closes, today_closes):
    d1 = dt.date(2026, 11, 23); d2 = dt.date(2026, 11, 24)
    return pd.concat([day_bars(d1, prev_closes), day_bars(d2, today_closes)])

def flat(n, v): return [v] * n

def test_pdh_break_alert_and_dedupe():
    prev = list(np.linspace(24900, 25100, 75))             # PDH ~ 25100+
    pdh = day_bars(dt.date(2026, 11, 23), prev)["High"].max()
    today = flat(10, 25050) + [pdh + 60] + flat(3, pdh + 70)
    df = frame(prev, today)
    # 11th bar starts 10:05, closes 10:10 -> run at 10:11
    st = {}
    now = dt.datetime(2026, 11, 24, 10, 11, tzinfo=A.IST)
    out = A.evaluate("NIFTY 50", df.iloc[:-3], st, now, CFG)
    assert any("previous-day high" in o for o in out)
    assert "Stop" in out[0] and "Wrong if" in out[0]
    out2 = A.evaluate("NIFTY 50", df.iloc[:-3], st, now, CFG)          # same run again
    assert out2 == []
    later = dt.datetime(2026, 11, 24, 10, 26, tzinfo=A.IST)
    assert not [o for o in A.evaluate("NIFTY 50", df, st, later, CFG) if "previous-day high" in o]

def test_no_alert_before_0930_and_gap_up_is_not_a_break():
    prev = flat(75, 25000)
    df = frame(prev, [25000, 25000, 25100, 25100, 25100, 25100])   # cross at 9:25 candle close 9:30? bar idx2 closes 9:30
    now = dt.datetime(2026, 11, 24, 9, 31, tzinfo=A.IST)
    out = A.evaluate("NIFTY 50", df.iloc[:6], {}, dt.datetime(2026, 11, 24, 9, 46, tzinfo=A.IST), CFG)
    assert [o for o in out if "previous-day" in o] == []            # gapped above PDH and stayed: not a break
    df2 = frame(prev, [25000, 25200, 25200])
    now = dt.datetime(2026, 11, 24, 9, 26, tzinfo=A.IST)
    assert A.evaluate("NIFTY 50", df2.iloc[:2], {}, now, CFG) == []   # closes 9:25 -> inside first 15 min

def test_rearm_after_gap():
    prev = flat(75, 25000)
    pdh = day_bars(dt.date(2026, 11, 23), prev)["High"].max()
    up = pdh * 1.003
    today = flat(10, 25000) + [up] + [25000] * 8 + [up] + [25000] + [up] + [up]
    df = frame(prev, today)
    st = {}
    fired = 0
    for k in range(10, len(today) + 1):
        now = dt.datetime.combine(dt.date(2026, 11, 24), dt.time(9, 15)) + pd.Timedelta(minutes=5 * k + 1)
        fired += len([o for o in A.evaluate("NIFTY 50", df.iloc[:75 + k], st, now.replace(tzinfo=A.IST), CFG) if "previous-day high" in o])
    # cross at idx10 (10:05-10:10) fires; idx19 cross is 45 min later -> fires; idx21 cross 10 min later -> suppressed
    assert fired == 2

def test_orb_and_commodity_has_no_orb():
    prev = flat(75, 25000)
    today = flat(6, 25000) + [25010] * 4 + [25060]
    df = frame(prev, today)
    pre_hi = df[df.index.date == dt.date(2026, 11, 24)].iloc[:6]["High"].max()
    now = dt.datetime(2026, 11, 24, 10, 11, tzinfo=A.IST)
    out = A.evaluate("NIFTY 50", df, {}, now, CFG)
    assert any("broke above opening range high" in o for o in out)
    out = A.evaluate("GOLD (COMEX)", df, {}, now, CFG)
    assert not any("opening range" in o for o in out)

def test_custom_level_and_cap():
    prev = flat(75, 25000)
    df = frame(prev, flat(12, 25000) + [25120] + [25120])
    lv = [{"instrument": "NIFTY 50", "level": 25100, "direction": "above", "note": "mine"}]
    now = dt.datetime(2026, 11, 24, 10, 21, tzinfo=A.IST)
    out = A.evaluate("NIFTY 50", df.iloc[:-1], {}, now, CFG, lv)
    assert any("your level" in o and "mine" in o for o in out)
    cfg = {**CFG, "daily_cap": 0}
    assert A.evaluate("NIFTY 50", df.iloc[:-1], {}, now, cfg, lv) == []

def test_supertrend_flip_down():
    prev = list(np.linspace(24800, 25200, 75))
    today = list(np.linspace(25200, 25250, 12)) + list(np.linspace(25250, 24700, 24))
    df = frame(prev, today)
    st, n = {}, 0
    for k in range(12, len(today) + 1):
        now = dt.datetime.combine(dt.date(2026, 11, 24), dt.time(9, 15)) + pd.Timedelta(minutes=5 * k + 1)
        n += len([o for o in A.evaluate("BANK NIFTY", df.iloc[:75 + k], st, now.replace(tzinfo=A.IST), CFG) if "Supertrend flipped DOWN" in o])
    assert n >= 1

def test_stale_alert_dropped_and_delayed_tag():
    prev = flat(75, 25000)
    pdh = day_bars(dt.date(2026, 11, 23), prev)["High"].max()
    df = frame(prev, flat(10, 25000) + [pdh * 1.003])
    late = dt.datetime(2026, 11, 24, 10, 20, tzinfo=A.IST)               # closed 10:10 -> 10 min late
    out = A.evaluate("NIFTY 50", df, {}, late, CFG)
    assert out and out[0].startswith("(delayed")
    very = dt.datetime(2026, 11, 24, 11, 30, tzinfo=A.IST)
    assert A.evaluate("NIFTY 50", df, {}, very, CFG) == []

def test_morning_brief_and_main_dry(tmp_path, monkeypatch):
    prev = list(np.linspace(24900, 25100, 75))
    df = frame(prev, flat(3, 25000))
    daily = pd.DataFrame({"Close": np.linspace(24000, 25000, 120)})
    now = dt.datetime(2026, 11, 24, 9, 12, tzinfo=A.IST)
    txt = A.morning_brief({"NIFTY 50": df}, {"NIFTY 50": daily}, now, CFG)
    assert "Morning brief" in txt and "Bullish" in txt and "NIFTY 50" in txt
    monkeypatch.setattr(A, "fetch", lambda t, i="5m", p="5d": df if i == "5m" else daily.set_index(pd.date_range("2025-01-01", periods=120)))
    st = tmp_path / "s.json"
    assert A.main(["--state", str(st), "--dry", "--now", "2026-11-24T09:12:00"]) == 0
    assert json.loads(st.read_text())["morning"] == "2026-11-24"
    assert A.main(["--state", str(st), "--dry", "--test"]) == 0
