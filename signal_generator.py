# llm/signal_generator.py
# Layer 3: LLM-based signal generator
#
# Takes current market state (OHLCV + indicators) and asks an LLM
# to reason about it and produce a structured trading signal.
#
# Supports two backends:
#   "claude"  → Claude API (fast, no GPU needed, costs ~$0.001/call)
#   "ollama"  → Local model via Ollama (free, needs a decent GPU or M-chip Mac)
#
# The key insight: LLMs are NOT magic predictors.
# They're used here for CONTEXTUAL REASONING — given indicators + recent
# price action, does the setup look clean? Is anything contradicting the signal?
# Think of it as a second opinion, not a crystal ball.

import json
import re
from dataclasses import dataclass
from typing import Literal
import pandas as pd


# ── Signal data structure ─────────────────────────────────────────────────────

@dataclass
class TradingSignal:
    action:      Literal["BUY", "SELL", "HOLD"]
    confidence:  float          # 0.0 to 1.0
    reasoning:   str            # LLM's explanation
    risk_level:  Literal["LOW", "MEDIUM", "HIGH"]
    key_factors: list[str]      # top 3 factors driving the signal
    suggested_stop: float       # suggested stop loss price
    suggested_target: float     # suggested take profit price

    def __str__(self):
        bar = "█" * int(self.confidence * 10) + "░" * (10 - int(self.confidence * 10))
        return (
            f"\n  ┌─ LLM Signal ─────────────────────────────────────\n"
            f"  │  Action:     {self.action}\n"
            f"  │  Confidence: {bar} {self.confidence:.0%}\n"
            f"  │  Risk level: {self.risk_level}\n"
            f"  │  Stop:       {self.suggested_stop:.1f}\n"
            f"  │  Target:     {self.suggested_target:.1f}\n"
            f"  │  Factors:\n"
            + "".join(f"  │    • {f}\n" for f in self.key_factors) +
            f"  │  Reasoning:\n  │    {self.reasoning[:200]}{'...' if len(self.reasoning)>200 else ''}\n"
            f"  └──────────────────────────────────────────────────"
        )


# ── Market context builder ────────────────────────────────────────────────────

def build_market_context(df: pd.DataFrame, lookback: int = 10) -> str:
    """
    Summarise recent market state into a text prompt for the LLM.
    Uses the last `lookback` rows of your featured DataFrame.
    """
    recent = df.tail(lookback)
    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    # Price action summary
    close     = latest["close"]
    prev_close= prev["close"]
    chg_pct   = (close - prev_close) / prev_close * 100
    direction = "up" if chg_pct > 0 else "down"

    # Recent range
    period_high = recent["high"].max()
    period_low  = recent["low"].min()

    # Trend context
    trend = "BULLISH" if latest.get("trend", 1) == 1 else "BEARISH"
    regime = latest.get("regime", "unknown").upper()

    lines = [
        f"INSTRUMENT: Nifty 50 Index",
        f"TIMESTAMP: {df.index[-1].strftime('%Y-%m-%d %H:%M')} IST",
        f"",
        f"PRICE ACTION (last {lookback} bars):",
        f"  Current close:    {close:.2f}",
        f"  Previous close:   {prev_close:.2f}  ({chg_pct:+.2f}% {direction})",
        f"  {lookback}-bar high:  {period_high:.2f}",
        f"  {lookback}-bar low:   {period_low:.2f}",
        f"",
        f"TREND INDICATORS:",
        f"  EMA9:   {latest['ema9']:.2f}",
        f"  EMA21:  {latest['ema21']:.2f}",
        f"  EMA50:  {latest.get('ema50', 'N/A')}",
        f"  EMA200: {latest.get('ema200', 'N/A')}",
        f"  Trend:  {trend}  |  Regime: {regime}",
        f"",
        f"MOMENTUM:",
        f"  RSI(14):      {latest['rsi14']:.1f}",
        f"  MACD:         {latest['macd']:.2f}",
        f"  MACD Signal:  {latest['macd_signal']:.2f}",
        f"  MACD Hist:    {latest['macd_hist']:.2f}  "
            f"({'positive' if latest['macd_hist'] > 0 else 'negative'})",
        f"",
        f"VOLATILITY:",
        f"  ATR(14):      {latest['atr14']:.2f}  ({latest.get('atr_pct', 0):.2f}% of price)",
        f"  BB Upper:     {latest['bb_upper']:.2f}",
        f"  BB Mid:       {latest['bb_mid']:.2f}",
        f"  BB Lower:     {latest['bb_lower']:.2f}",
        f"  BB %position: {latest['bb_pct']:.2f}  (0=at lower, 1=at upper)",
        f"",
        f"VOLUME:",
        f"  Volume ratio: {latest.get('vol_ratio', 1.0):.2f}x  "
            f"({'above' if latest.get('vol_ratio', 1) > 1 else 'below'} average)",
    ]

    # Add recent price history as a mini table
    lines.append(f"\nRECENT PRICE HISTORY:")
    for idx, row in recent.tail(5).iterrows():
        daily_chg = row.get("daily_return", 0)
        lines.append(f"  {idx.strftime('%Y-%m-%d')}  C={row['close']:.1f}  "
                     f"RSI={row['rsi14']:.0f}  "
                     f"Chg={daily_chg:+.2f}%")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are a quantitative trading analyst specialising in Indian equity indices (Nifty 50).

Your role:
- Analyse the provided market data and technical indicators
- Produce a structured trading signal: BUY, SELL, or HOLD
- Be conservative — only signal BUY/SELL when the evidence is clear
- HOLD is the right answer when signals are mixed or risk is high

Risk management rules you must follow:
- Never signal against a strong trend (e.g. BUY when price is below EMA200)
- Flag HIGH risk when ATR% > 1.5% (volatile — wide stops needed)
- Flag HIGH risk when RSI > 75 or RSI < 25 (extremes)
- Only suggest BUY when RSI is between 35–65 (not chasing overbought)

You MUST respond in this exact JSON format, nothing else:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": <float 0.0-1.0>,
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "key_factors": ["factor1", "factor2", "factor3"],
  "suggested_stop": <float — price level for stop loss>,
  "suggested_target": <float — price level for take profit>,
  "reasoning": "<2-3 sentence explanation>"
}"""


def build_user_prompt(market_context: str) -> str:
    return (
        f"Analyse the following Nifty 50 market data and provide your trading signal.\n\n"
        f"{market_context}\n\n"
        f"Respond with JSON only. No preamble, no markdown fences."
    )


# ── Claude API backend ────────────────────────────────────────────────────────

def get_signal_claude(api_key: str, market_context: str) -> TradingSignal:
    """Generate signal using Claude API (remote)."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(market_context)}],
    )
    
    raw = response.content[0].text.strip()
    return _parse_signal_response(raw, market_context)


# ── Ollama (local) backend ────────────────────────────────────────────────────

def get_signal_ollama(
    market_context: str,
    model: str = "llama3",
    base_url: str = "http://localhost:11434"
) -> TradingSignal:
    """
    Generate signal using a local model via Ollama.
    
    Setup:
        1. Install Ollama: https://ollama.com
        2. Pull a model: ollama pull llama3
           (alternatives: mistral, phi3, gemma2 — smaller = faster)
        3. Run Ollama: ollama serve  (or it auto-starts)
    
    Note: Local models are slower and less precise than Claude for
    structured JSON output. Add retry logic for production use.
    """
    import urllib.request

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(market_context)},
        ],
        "stream": False,
        "format": "json",   # Ollama's native JSON mode — helps a lot
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    
    raw = data["message"]["content"].strip()
    return _parse_signal_response(raw, market_context)


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_signal_response(raw: str, market_context: str) -> TradingSignal:
    """
    Parse LLM JSON response into a TradingSignal.
    Falls back to HOLD if parsing fails (fail-safe default).
    """
    # Strip any markdown fences the model may have added despite instructions
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    
    try:
        data = json.loads(raw)
        
        return TradingSignal(
            action         = data.get("action", "HOLD").upper(),
            confidence     = float(data.get("confidence", 0.5)),
            reasoning      = data.get("reasoning", "No reasoning provided"),
            risk_level     = data.get("risk_level", "MEDIUM").upper(),
            key_factors    = data.get("key_factors", [])[:3],
            suggested_stop = float(data.get("suggested_stop", 0)),
            suggested_target = float(data.get("suggested_target", 0)),
        )
    
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  Warning: LLM response parse error ({e}). Defaulting to HOLD.")
        print(f"  Raw response: {raw[:200]}")
        
        # Safe fallback
        latest_price = _extract_price_from_context(market_context)
        return TradingSignal(
            action="HOLD", confidence=0.0, risk_level="HIGH",
            reasoning=f"Parse error — defaulting to HOLD. Raw: {raw[:100]}",
            key_factors=["Parse error", "Defaulting to safe HOLD"],
            suggested_stop=latest_price * 0.985,
            suggested_target=latest_price * 1.015,
        )


def _extract_price_from_context(context: str) -> float:
    """Extract current close price from context string for fallback."""
    match = re.search(r"Current close:\s+([\d.]+)", context)
    return float(match.group(1)) if match else 20000.0


# ── Main signal entry point ───────────────────────────────────────────────────

def generate_signal(
    df: pd.DataFrame,
    mode: str = "claude",
    api_key: str = None,
    ollama_model: str = "llama3",
    ollama_url: str = "http://localhost:11434",
) -> TradingSignal:
    """
    Generate a trading signal from the latest data in df.
    
    Args:
        df:           DataFrame with OHLCV + features (output of add_features())
        mode:         "claude" or "ollama"
        api_key:      Anthropic API key (required for claude mode)
        ollama_model: model name (for ollama mode)
        ollama_url:   Ollama server URL (for ollama mode)
    
    Returns:
        TradingSignal with action, confidence, reasoning, stop, target
    """
    context = build_market_context(df)
    
    if mode == "claude":
        if not api_key:
            raise ValueError("api_key required for claude mode")
        return get_signal_claude(api_key, context)
    
    elif mode == "ollama":
        return get_signal_ollama(context, model=ollama_model, base_url=ollama_url)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'claude' or 'ollama'")


# ── Demo with synthetic data ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.kite_data import make_synthetic_nifty, add_features

    print("LLM Signal Generator — demo with synthetic data\n")
    
    df = make_synthetic_nifty(periods=300)
    df = add_features(df)
    
    print("Market context that will be sent to LLM:")
    print("-" * 60)
    ctx = build_market_context(df)
    print(ctx)
    print("-" * 60)
    
    # To test with real LLM, uncomment one of:
    # signal = generate_signal(df, mode="claude", api_key="YOUR_KEY")
    # signal = generate_signal(df, mode="ollama", ollama_model="llama3")
    
    print("\nTo generate a real signal:")
    print("  Claude: signal = generate_signal(df, mode='claude', api_key='sk-...')")
    print("  Ollama: signal = generate_signal(df, mode='ollama', ollama_model='llama3')")
    print("\nMake sure Ollama is running: ollama serve && ollama pull llama3")
