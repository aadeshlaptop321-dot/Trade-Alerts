# Breakout alerts (Telegram)

Checks Nifty, Bank Nifty, Sensex, Gold, Silver, Crude, Nat Gas on 5-minute closed candles and
messages you on Telegram. Edit `levels.json` for your own levels, `config.json` for settings.

## One-time setup
1. **Telegram bot**: in Telegram open @BotFather -> /newbot -> copy the token. Send any message to your new bot.
   Open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and copy `"chat":{"id": NUMBER`.
2. **GitHub repo** (public, so Actions minutes are unlimited): upload all these files, keep `.github/workflows/alerts.yml`.
   Settings -> Secrets and variables -> Actions -> add `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`.
3. **Test**: Actions tab -> alerts -> Run workflow -> tick "Send a test Telegram message".
4. **Exact timer** (GitHub's own timer runs late): GitHub -> Settings -> Developer settings -> Fine-grained tokens
   -> only this repo -> permission *Actions: Read and write*. Then at cron-job.org create a job:
   - URL `https://api.github.com/repos/<you>/<repo>/actions/workflows/alerts.yml/dispatches`
   - Method POST, body `{"ref":"main"}`
   - Headers: `Authorization: Bearer <token>`, `Accept: application/vnd.github+json`, `User-Agent: alerts`
   - Schedule: time zone Asia/Kolkata, Mon-Fri, hours 9-23, minutes 1,6,11,16,21,26,31,36,41,46,51,56.

## Custom levels (`levels.json`)
`{"instrument": "NIFTY 50", "level": 26100, "direction": "above", "note": "my breakout"}`
Instruments: NIFTY 50, BANK NIFTY, SENSEX, GOLD (COMEX), SILVER (COMEX), CRUDE OIL (WTI), NATURAL GAS.

Not financial advice - alerts are rule-based and can be wrong. Run tests: `pip install pytest && pytest`.
