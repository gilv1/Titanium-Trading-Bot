# 🏆 Titanium Warrior v3 — Multi-Engine Trading Bot

> Converting $500 to $15,000 using 4 independent trading engines with an AI-powered, self-learning decision brain, compound auto-scaling, and real-time Telegram notifications.

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                     TITANIUM WARRIOR v3                          ║
║                                                                  ║
║  🔵 Motor 1: FUTURES (MNQ/NQ) — Margin Account, 23h/day        ║
║  🟢 Motor 2: OPTIONS 0DTE (SPY/QQQ) — Cash Account (off)       ║
║  🟡 Motor 3: MOMO SMALL-CAPS — Cash Account (off)               ║
║  🟠 Motor 4: CRYPTO (BTC/ETH via IBKR) — Margin Account        ║
║                                                                  ║
║  🧠 AI Brain — AUTOEVOLUTIVE, learns from every trade           ║
║  💰 Reto Tracker — $500→$15,000 compound auto-scale             ║
║  📱 Telegram — Real-time trade notifications                     ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Features

| Component | Description |
|-----------|-------------|
| **4 Engines** | Futures, 0DTE Options, MoMo small-caps, Crypto — each running independently via `asyncio` |
| **AI Brain** | 0–100 trade score, self-learning win rates per setup/session/hour/volatility |
| **Reto Tracker** | Phase-based compound scaling ($500→$15k in 4 phases) |
| **Risk Manager** | Kill switch, engine pause, correlation guard, PDT compliance |
| **MoMo Scanner** | Pre-market gap scanner with 0–110 scoring, news catalyst validation |
| **Telegram** | Entry/exit alerts, daily summary, phase changes, milestone alerts |
| **Trade Journal** | CSV + JSON logs with win-rate and P&L analytics |
| **Backtester** | Walk-forward simulation on historical 1-min CSV data |

---

## Project Structure

```
Titanium-Trading-Bot/
├── .env.example              ← Copy to .env and fill in your credentials
├── .gitignore
├── requirements.txt
├── README.md
├── main.py                   ← Entry point
├── backtest.py               ← Backtesting framework
│
├── config/
│   └── settings.py           ← All env variables, phases, sessions
│
├── core/
│   ├── brain.py              ← AI Brain (AUTOEVOLUTIVE scoring + learning)
│   ├── reto_tracker.py       ← Compound interest & phase management
│   ├── risk_manager.py       ← Global risk veto layer
│   └── connection.py         ← IBKR dual-account connection manager
│
├── engines/
│   ├── base_engine.py        ← Abstract base + shared dataclasses
│   ├── futures_engine.py     ← Motor 1: MNQ/NQ
│   ├── options_engine.py     ← Motor 2: 0DTE SPY/QQQ (disabled)
│   ├── momo_engine.py        ← Motor 3: MoMo small-caps (disabled)
│   └── crypto_engine.py      ← Motor 4: BTC/ETH via IBKR
│
├── analysis/
│   ├── technical.py          ← VWAP, EMA, RSI, MACD, ATR, RVOL, BB
│   ├── patterns.py           ← Pattern detectors returning Signal objects
│   └── scanner.py            ← Pre-market MoMo scanner (0–110 scoring)
│
├── notifications/
│   └── telegram.py           ← Async Telegram notifier
│
├── data/
│   └── news.py               ← Alpha Vantage news client with cache + rate limit
│
├── journal/
│   └── trade_journal.py      ← CSV + JSON trade logging & analytics
│
└── tests/
    ├── test_brain.py
    ├── test_risk_manager.py
    ├── test_reto_tracker.py
    └── test_scanner.py
```

---

## Prerequisites

- **Python 3.11+**
- **IBKR TWS or IB Gateway** (running locally, paper trading recommended to start)
- **Telegram Bot** (optional but highly recommended)
- **Alpha Vantage API key** (free tier: 5 calls/min, 500/day — for news & momo data)

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/gilv1/Titanium-Trading-Bot.git
cd Titanium-Trading-Bot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Edit .env with your IBKR accounts, API keys, Telegram credentials

# 5. Run in paper trading mode (default)
python main.py --paper

# 6. Run backtest
python backtest.py
```

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and fill in your values:

```bash
# IBKR Connection
IBKR_HOST=127.0.0.1
IBKR_PORT=7497          # TWS paper: 7497 | Gateway paper: 4002
IBKR_MARGIN_ACCOUNT=DU123456   # Your paper trading account ID
IBKR_CASH_ACCOUNT=DU654321     # For cash account (options/momo)

# API Keys
ALPHA_VANTAGE_API_KEY=demo     # Get free key at alphavantage.co

# Telegram (optional but recommended)
TELEGRAM_BOT_TOKEN=123456:ABC-DEF
TELEGRAM_CHAT_ID=-100123456789

# Engine Toggles
ENABLE_FUTURES=true
ENABLE_OPTIONS=false
ENABLE_MOMO=false
ENABLE_CRYPTO=true

# Risk
INITIAL_CAPITAL=500
MAX_DAILY_RISK_PCT=8
KILL_SWITCH_PCT=12

# Mode
TRADING_MODE=paper
```

### ⚠️ Security

- `.env` is in `.gitignore` — **never commit it**
- Use `python-dotenv` (already in requirements) to load credentials
- Never hardcode API keys or account numbers in source code

---

## Setting Up IBKR Paper Trading

1. Download [Interactive Brokers TWS](https://www.interactivebrokers.com/en/trading/tws.php) or [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
2. Log in with your IBKR credentials and select **Paper Trading** account
3. In TWS: **Edit → Global Configuration → API → Settings**
   - Enable "Enable ActiveX and Socket Clients"
   - Set Socket Port: `7497`
   - Enable "Allow connections from localhost only"
4. Note your paper trading account ID (starts with `DU`)
5. Set `IBKR_MARGIN_ACCOUNT=DU123456` in `.env`

---

## Setting Up Telegram Notifications

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Copy the **bot token** (format: `123456789:ABCdef...`)
4. Set `TELEGRAM_BOT_TOKEN=<your token>` in `.env`
5. Start a conversation with your bot (or add it to a group)
6. To get your chat ID, send a message then visit:  
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
7. Copy the `chat.id` value and set `TELEGRAM_CHAT_ID=<your id>` in `.env`

---

## Engine Descriptions

### 🔵 Motor 1: Futures (MNQ / NQ)

Trades Micro E-mini Nasdaq-100 (MNQ) and E-mini Nasdaq-100 (NQ) via CME.
Operates during Tokyo, London, and NY sessions (phase-dependent).

**5 Setups:**
| Setup | Trigger |
|-------|---------|
| **VWAP Bounce** | Price touches VWAP, RSI 35–45 (long) or 55–65 (short), decreasing volume |
| **ORB** | First 15-min range breakout with volume >150% avg, EMA 9 aligned |
| **EMA 9/21 Pullback** | Trend confirmed by EMA cross, price pulls to EMA 9, MACD positive |
| **Liquidity Grab** | Price spikes beyond S/R by 2–5 pts then reverses with explosive volume |
| **News Burst** | Major macro event — wait for first pullback after initial move |

**Position Management:**
- Bracket orders (entry + SL + TP)
- Sell 50 % at Target 1
- At +15 pts: move SL to breakeven + 1
- At +25 pts: trail by 8 pts

---

### 🟢 Motor 2: 0DTE Options *(disabled by default)*

Buys same-day-expiry calls/puts on SPY and QQQ.
- Stop loss: −30 % of premium
- Take profit: +80 % to +150 %
- Theta guard: close if no movement within 3–5 minutes

Enable with `ENABLE_OPTIONS=true` in `.env`.

---

### 🟡 Motor 3: MoMo Small-Caps *(disabled by default)*

Scans pre-market for gap-up small-cap stocks with news catalysts.

**Hard Filters (ALL must pass):**
- Gap ≥ +10 %
- Float ≤ 10 M shares
- Price ≤ $20
- Verifiable news catalyst
- RVOL ≥ 5×

**Scoring (0–110):** RVOL, float size, price range, 52-week high, daily chart quality, sector momentum, pre-market volume, short interest, bid/ask spread.

**Entry Types:** Pullback to VWAP/EMA9, Dip Buy, Breakout

PDT compliant: max 3 day trades per rolling 5 business days.

Enable with `ENABLE_MOMO=true` in `.env`.

---

### 🟠 Motor 4: Crypto (BTC / ETH)

Trades Bitcoin and Ethereum through IBKR Paxos — keeping everything on one platform.
- 24/7 operation with Asia/Europe/US session tracking
- SL: 1.5–2 % of position value
- TP: 3–5 % (min 1:2 R:R)
- Correlation guard: no BTC long when NQ long (correlation ~0.7)

---

## Phase System

| Phase | Capital Range | Instrument | Contracts | Sessions |
|-------|--------------|------------|-----------|----------|
| 1 | $500 – $1,500 | MNQ | 1 | NY only |
| 2 | $1,500 – $4,500 | MNQ | 3 | London + NY |
| 3 | $4,500 – $9,000 | NQ | 1 | Tokyo + London + NY |
| 4 | $9,000 – $15,000 | NQ | 2 | Tokyo + London + NY |

**Capital Milestone Alerts:**
- $2,500 → Alert to withdraw $500 (original capital recovery)
- $7,500 → Alert to move $2,000 to a safe account
- $12,000 → Alert to move $3,000 to a safe account

---

## Risk Management Rules

| Rule | Value |
|------|-------|
| Max daily loss (global) | 8 % of capital |
| Max daily loss per engine | Futures: 4 %, Others: 2 % |
| Kill switch threshold | 12 % daily loss — shuts all engines for 24 h |
| Consecutive loss pause | 3 losses → engine paused for 4 h |
| Max simultaneous positions | 3 |
| Max trades/day | Phase 1–2: 6/engine, Phase 3–4: 8/engine |
| PDT (momo) | Max 3 day trades per rolling 5 business days |
| Correlation guard | Long NQ + Long BTC blocked (corr ~0.7) |

---

## AI Brain — AUTOEVOLUTIVE

The brain scores every potential trade 0–100:

| Component | Max Score |
|-----------|-----------|
| Pattern confidence | 25 |
| Session win-rate history | 20 |
| ATR / volatility assessment | 15 |
| Daily drawdown status | 15 |
| Correlation / position count | 15 |
| Trend alignment | 10 |

**Decision thresholds:**
- Score > 75 → Full size (1.0×)
- Score > 65 → Half size (0.5×)
- Score ≤ 65 → No trade

After each closed trade the brain updates win rates per: setup type, session, day of week, hour of day, and volatility regime. If a pattern loses 3 times in a row, confidence is penalised until the pattern wins again (regime change signal).

Memory is saved to `data/brain_memory.json` (excluded from git).

---

## Running

```bash
# Paper trading (default)
python main.py --paper

# Live trading
python main.py --live

# Backtesting (place CSV files in data/historical/)
python backtest.py
```

---

## Backtesting

1. Create the `data/historical/` directory
2. Add 1-minute OHLCV CSV files named by ticker (e.g. `MNQ.csv`)
3. Required columns: `time,open,high,low,close,volume`
4. Run: `python backtest.py`
5. Results saved to `journal/backtest_results.json`

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ib_insync not installed` | `pip install ib_insync` |
| IBKR connection refused | Ensure TWS/Gateway is running on port 7497, API enabled |
| Telegram not sending | Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` |
| `ALPHA_VANTAGE_API_KEY not set` | Get a free key at [alphavantage.co](https://www.alphavantage.co) |
| Brain memory not loading | Delete `data/brain_memory.json` to reset (file may be corrupt) |
| Import errors | Run `pip install -r requirements.txt` in your virtual environment |

---

## Disclaimer

⚠️ **Trading involves substantial risk of loss. This software is for educational and informational purposes only. Past performance is not indicative of future results. Never trade with money you cannot afford to lose. The authors are not responsible for any financial losses incurred using this software. Always start with paper trading.**
