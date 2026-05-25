# Nifty 50 Algorithmic Trading System
# Quickstart guide and main runner

## Project structure

```
nifty_trader/
├── requirements.txt          # pip install -r requirements.txt
├── config/
│   └── settings.py           # API keys + trading parameters
├── data/
│   └── kite_data.py          # Layer 1: data download + feature engineering
├── strategies/
│   └── backtest.py           # Layer 2: backtrader strategies + backtesting
├── llm/
│   └── signal_generator.py   # Layer 3: Claude/Ollama signal generation
└── execution/
    └── paper_trader.py       # Layer 4: paper trading engine
```

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

For local LLM support:
```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3       # ~4GB, good quality
# or:
ollama pull mistral      # ~4GB, slightly faster
# or:
ollama pull phi3         # ~2GB, smallest/fastest
```

---

## Step 2 — Set up Zerodha Kite API

1. Go to https://developers.kite.trade/ and create an app
2. Note your `api_key` and `api_secret`
3. Update `config/settings.py` with your keys
4. Enable Kite Connect subscription (₹2000/month) in your Zerodha account

---

## Step 3 — Download historical data

```python
from kiteconnect import KiteConnect
from data.kite_data import get_kite_session, download_nifty_dataset

# One-time login (opens browser, get request_token from redirect URL)
kite = get_kite_session(API_KEY, API_SECRET, request_token="TOKEN_FROM_URL")
# Save the access_token — valid until midnight IST

# Download 3 years of Nifty 50 data (daily + hourly + 15min)
datasets = download_nifty_dataset(kite, years_back=3)
# Saves to ./data/historical/ as Parquet files
```

---

## Step 4 — Run backtests

```python
from data.kite_data import load_historical, add_features
from strategies.backtest import MomentumStrategy, MeanReversionStrategy, run_backtest

df = load_historical("NIFTY50", interval="1day")
df = add_features(df)

# Test momentum strategy
metrics = run_backtest(df, MomentumStrategy, initial_capital=500_000)

# Test mean reversion
metrics = run_backtest(df, MeanReversionStrategy, initial_capital=500_000)
```

Or test immediately on synthetic data (no Kite needed):
```bash
cd nifty_trader
python strategies/backtest.py
```

---

## Step 5 — Test LLM signal generation

```python
from data.kite_data import load_historical, add_features
from llm.signal_generator import generate_signal

df = load_historical("NIFTY50", interval="1day")
df = add_features(df)

# Using Claude API
signal = generate_signal(df, mode="claude", api_key="YOUR_ANTHROPIC_KEY")
print(signal)

# Using local Ollama
signal = generate_signal(df, mode="ollama", ollama_model="llama3")
print(signal)
```

---

## Step 6 — Paper trade live

```python
from execution.paper_trader import PaperTrader, start_live_paper_trading

trader = PaperTrader(
    capital=500_000,
    risk_pct=1.0,           # 1% max risk per trade
    llm_mode="claude",
    llm_api_key="YOUR_KEY",
)

# Run during market hours (9:15 AM – 3:30 PM IST)
start_live_paper_trading(
    kite=kite,
    trader=trader,
    strategy_class=MomentumStrategy,
    instrument_token=256265,  # Nifty 50 spot
    interval="15min",
    run_hours=7,
)

# Check performance anytime
trader.print_summary()
```

---

## What to look for before going live

Minimum criteria after at least 30 paper trades:

| Metric            | Minimum threshold |
|-------------------|-------------------|
| Win rate          | > 50%             |
| Sharpe ratio      | > 1.0             |
| Max drawdown      | Acceptable to you |
| Profit factor     | > 1.5             |
| LLM avg confidence| > 0.65            |

---

## Important notes

- **Paper trade for at least 2–3 months** before considering real money
- Start with daily bars — easier to reason about than intraday
- The LLM is a *filter*, not a predictor. It helps avoid low-quality setups.
- F&O (futures) have leverage — ₹1L capital controls a much larger position.
  The risk manager handles this, but understand your actual exposure.
- Keep position size at 1% risk. At 20 trades, worst case = 20% drawdown.
  That is real money. Respect it.
