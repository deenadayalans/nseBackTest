# NSE OEH Trading System — Startup Guide

Everything you need to get running on a new machine, from scratch.

---

## What This System Does

Trades the **Open = High / Open = Low (OEH/OEL) reversal pattern** on Nifty and Sensex index options.

**Strategy logic:**
- Market opens, spot price hits a high (Open = High) then pulls back
- The pullback is temporary — most of the time price reverses back to the OEH level before noon
- We buy the ATM Call option at the reversal point and ride it back up
- Target: +55% on premium | Stop: -50% on premium

**Instruments traded:**
| Instrument | Days | Expiry |
|---|---|---|
| Nifty 50 | Monday + Tuesday | Tuesday weekly |
| Sensex | Wednesday + Thursday | Thursday weekly |

**Filters before taking a trade:**
1. Daily SuperTrend = BUY (macro trend is up)
2. Spot price > 20 DMA (price above 20-day moving average)
3. Spot price > 50 DMA (medium-term uptrend)
4. 5-min SuperTrend = BUY on the bar before OEH
5. Spot close > EMA-50 on 5-min chart
6. Spot drops ≥ 0.4% from OEH before reversing
7. 2 consecutive green (higher-close) bars = reversal confirmed
8. OEH window: 9:15 AM – 12:30 PM only

---

## One-Time Setup (New Machine)

### 1. Install Python 3.11–3.13

**Mac:**
```bash
brew install python@3.13
```

**Windows:** Download from [python.org](https://python.org)

Verify:
```bash
python3 --version   # should show 3.11.x, 3.12.x or 3.13.x
```

---

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/nseBacktest.git
cd nseBacktest
```

---

### 3. Create Virtual Environment and Install Packages

```bash
python3 -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

---

### 4. Create Your `.env` File

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```ini
# ── Zerodha Kite Connect ───────────────────────────────────────────
# Get from: https://developers.kite.trade → My Apps → Create App
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_ACCESS_TOKEN=          # leave blank — filled automatically each morning

# ── Telegram alerts ────────────────────────────────────────────────
# Step 1: Open Telegram → search @BotFather → /newbot → copy the token
# Step 2: Open your bot → press Start
# Step 3: Visit https://api.telegram.org/bot<TOKEN>/getUpdates
#         Find "chat": {"id": 123456789} — that is your TG_CHAT_ID
TG_BOT_TOKEN=123456789:ABCdef...
TG_CHAT_ID=741618245

# ── WhatsApp via CallMeBot (optional backup) ───────────────────────
# Save +34 644 59 77 31 as a contact → WhatsApp them:
#   "I allow callmebot to send me messages"
# They reply with your API key
WA_PHONE=+91XXXXXXXXXX
WA_APIKEY=

# ── Capital & sizing ───────────────────────────────────────────────
INITIAL_CAPITAL=200000      # ₹2 Lakhs
DEPLOY_PCT=0.20             # 20% of equity per trade
MAX_LOTS_NIFTY=10           # hard cap — prevents runaway sizing
MAX_LOTS_SENSEX=20
```

> **Kite redirect URL:** When creating your Kite app, set the redirect URL to:
> `http://localhost:8080/kite/callback`

---

### 5. Download Historical Spot Data (First Time Only)

This downloads Nifty and Sensex 5-min and daily candles going back ~3 years.
Takes about 5–10 minutes. Run once, then `start.py` keeps it updated daily.

```bash
python download_data.py
```

---

## Daily Usage (Every Trading Day)

**One command does everything:**

```bash
source venv/bin/activate      # skip if already active
python start.py
```

**What happens automatically:**

| Time | Action |
|---|---|
| On launch | Browser opens for Kite login. Paste the redirect URL. |
| 9:20 AM | Morning briefing sent — SuperTrend, 20 DMA, ATM strikes, market bias |
| 9:15–12:30 PM | OEH detection running on 5-min candles |
| Signal fires | Alert 1: "OEH spotted" (heads-up) |
| Reversal bar 1 | Alert 2: "🟢⬛ 1/2 bars" (progress) |
| Reversal bar 2 | Alert 3: Full entry signal with all CE/PE strikes, targets, stops |
| 3:30 PM | Daily P&L summary sent, alerter exits |
| After 3:30 PM | Spot data downloaded, options data downloaded, backtest re-runs |

---

## Alert Examples

**Morning briefing (9:20 AM):**
```
📊 NIFTY OEH WATCHLIST — 26 May 2026
⚡ EXPIRY DAY

Yesterday: 24,150  ▲0.42%

Direction indicators (daily chart):
  SuperTrend: 🟢 BUY
  20 DMA: 23,800  (above 20DMA ✅)
  50 DMA: 23,200  (above 50DMA ✅)

Overall bias: 🟢 Strong BUY — all 3 indicators bullish

📌 Expected ATM: 24150
🕘 OEH window: 09:15 – 12:30
```

**OEH detected (instant heads-up):**
```
👀 OEH SPOTTED — NIFTY
⏰ 09:20

Open = High = 24,150
Dropped to: 23,990  (160pts ↓0.66%)

Direction confidence: Strong ✅✅

⏳ Watching for reversal...
Will alert on each green bar → entry when confirmed 🔔
```

**Entry signal:**
```
🚀 ENTER NOW — NIFTY REVERSAL CONFIRMED
🟢🟢 2 green bars done!

OEH level: 24,150
Spot now:  24,020  (130pts below OEH)
Ride back to: ~24,150

CE strikes (buy for upside):
  23900: ₹180  T:₹279  S:₹90
  24000: ₹120  T:₹186  S:₹60 ✅ ◀ ATM
  24100: ₹72   T:₹112  S:₹36
  24200: ₹38   T:₹59   S:₹19

⚡ Enter at market — target +55%, stop -50%
```

---

## Running the Backtest

```bash
python oeh_reversal.py
```

Shows historical P&L, win rate, CAGR, max drawdown for Nifty and Sensex using your capital settings from `.env`.

---

## File Overview

| File | Purpose |
|---|---|
| `start.py` | **Main entry point** — runs everything in sequence |
| `live_alerter.py` | Real-time OEH detection + Telegram/WhatsApp alerts |
| `oeh_reversal.py` | Historical backtest engine |
| `trade_journal.py` | Logs every live signal to `journal/YYYY-MM-DD.csv` |
| `download_data.py` | Downloads Nifty/Sensex spot OHLCV to parquet files |
| `download_options.py` | Downloads real options OHLCV (`--today` for daily refresh) |
| `login.py` | Kite OAuth login (called by `start.py`) |
| `kite_data.py` | Kite API helpers, indicator calculations |
| `settings.py` | Loads all config from `.env` |
| `.env.example` | Template — copy to `.env` and fill in |

**Folders created automatically at runtime:**
| Folder | Contents |
|---|---|
| `data/historical/` | Spot OHLCV parquet files |
| `data/options/` | Real option OHLCV parquet files |
| `journal/` | Daily trade logs (CSV, one per day) |

---

## Troubleshooting

**`ModuleNotFoundError`**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**`KITE_ACCESS_TOKEN missing` / login errors**
```bash
python start.py     # or: python login.py
# Token expires at midnight IST — must re-login each morning
```

**No alerts received on Telegram**
1. Check `TG_BOT_TOKEN` and `TG_CHAT_ID` in `.env`
2. Make sure you pressed **Start** on your bot in Telegram
3. Test: visit `https://api.telegram.org/bot<TOKEN>/getMe` in browser — should return your bot info

**No trades in backtest**
- All filters may be too strict on the available data range
- Try setting `REQUIRE_DAILY_DMA=False` temporarily in `.env` (add the key)
- Check data exists: `ls data/historical/`

**`data/historical/` is empty on new machine**
```bash
python download_data.py     # run once after login to fetch historical spot data
```

---

## Position Sizing Logic

```
Deploy amount = Current equity × DEPLOY_PCT
Lots          = floor(Deploy / (option_price × lot_size))
Max lots      = MAX_LOTS_NIFTY or MAX_LOTS_SENSEX (hard cap)
```

**Example** — equity ₹2L, ATM option at ₹80, Nifty lot size 65:
```
Deploy = ₹2,00,000 × 20% = ₹40,000
Lots   = floor(40,000 / (80 × 65)) = 7 lots → 455 units
```

Sizing grows as equity grows (compounding) and shrinks on losses (automatic protection).

---

## Lot Sizes (as of 2026)

| Instrument | Lot Size |
|---|---|
| Nifty 50 | 65 units |
| Sensex | 20 units |

---

*Last updated: May 2026*
