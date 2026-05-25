# settings.py — all config loaded from .env (never hardcode secrets)
# Copy .env.example → .env and fill in your values before running anything.

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()   # reads .env from the project root


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass
class Config:
    # ── Zerodha Kite ──────────────────────────────────────────────
    KITE_API_KEY:    str = field(default_factory=lambda: _env("KITE_API_KEY"))
    KITE_API_SECRET: str = field(default_factory=lambda: _env("KITE_API_SECRET"))

    # Access token is refreshed daily via login.py — stored in .env at runtime
    KITE_ACCESS_TOKEN: str = field(default_factory=lambda: _env("KITE_ACCESS_TOKEN"))

    # ── LLM: Claude (remote) ─────────────────────────────────────
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))

    # ── LLM: Ollama (local) ───────────────────────────────────────
    OLLAMA_BASE_URL: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    OLLAMA_MODEL:    str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3"))

    # ── NSE instrument tokens ─────────────────────────────────────
    NIFTY_SPOT_TOKEN: int = 256265   # Nifty 50 spot — fixed by NSE/Kite, never changes
    DEFAULT_EXCHANGE: str = "NSE"

    # ── Risk parameters ───────────────────────────────────────────
    MAX_RISK_PER_TRADE_PCT:   float = 1.0
    MAX_OPEN_TRADES:          int   = 1
    STOP_LOSS_ATR_MULTIPLIER: float = 1.5

    # ── Trading capital & sizing ──────────────────────────────────
    INITIAL_CAPITAL:  float = field(default_factory=lambda: float(_env("INITIAL_CAPITAL",  "200000")))
    DEPLOY_PCT:       float = field(default_factory=lambda: float(_env("DEPLOY_PCT",        "0.20")))
    MAX_LOTS_NIFTY:   int   = field(default_factory=lambda: int(_env("MAX_LOTS_NIFTY",    "10")))
    MAX_LOTS_SENSEX:  int   = field(default_factory=lambda: int(_env("MAX_LOTS_SENSEX",   "20")))

    # ── Telegram alerts ───────────────────────────────────────────
    TG_BOT_TOKEN: str = field(default_factory=lambda: _env("TG_BOT_TOKEN"))
    TG_CHAT_ID:   str = field(default_factory=lambda: _env("TG_CHAT_ID"))

    # ── WhatsApp alerts (CallMeBot) ───────────────────────────────
    WA_PHONE:  str = field(default_factory=lambda: _env("WA_PHONE"))
    WA_APIKEY: str = field(default_factory=lambda: _env("WA_APIKEY"))

    # ── LLM mode ─────────────────────────────────────────────────
    LLM_MODE: str = field(default_factory=lambda: _env("LLM_MODE", "claude"))

    # ── Data storage ─────────────────────────────────────────────
    DATA_DIR: str = "./data/historical"
    DB_PATH:  str = "./data/trades.db"


settings = Config()
