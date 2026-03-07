"""
Settings for Titanium Warrior v3.

Loads all environment variables from .env and defines trading constants,
phases, session times, and risk limits.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────
# IBKR Connection
# ──────────────────────────────────────────────────────────────
IBKR_HOST: str = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT: int = int(os.getenv("IBKR_PORT", "7497"))
IBKR_MARGIN_ACCOUNT: str = os.getenv("IBKR_MARGIN_ACCOUNT", "")
IBKR_CASH_ACCOUNT: str = os.getenv("IBKR_CASH_ACCOUNT", "")

# ──────────────────────────────────────────────────────────────
# API Keys
# ──────────────────────────────────────────────────────────────
ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# External scanner APIs (all free, used by ScannerPool for trade confirmation)
COINCAP_API_KEY: str = os.getenv("COINCAP_API_KEY", "")
COINGECKO_API_KEY: str = os.getenv("COINGECKO_API_KEY", "")
FREECRYPTO_API_KEY: str = os.getenv("FREECRYPTO_API_KEY", "")

# ──────────────────────────────────────────────────────────────
# Telegram
# ──────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ──────────────────────────────────────────────────────────────
# Engine Toggles
# ──────────────────────────────────────────────────────────────
ENABLE_FUTURES: bool = os.getenv("ENABLE_FUTURES", "true").lower() == "true"
ENABLE_OPTIONS: bool = os.getenv("ENABLE_OPTIONS", "false").lower() == "true"
ENABLE_MOMO: bool = os.getenv("ENABLE_MOMO", "false").lower() == "true"
ENABLE_CRYPTO: bool = os.getenv("ENABLE_CRYPTO", "false").lower() == "true"

# ──────────────────────────────────────────────────────────────
# Capital Allocation (percentages, must sum to 100)
# ──────────────────────────────────────────────────────────────
FUTURES_ALLOCATION: int = int(os.getenv("FUTURES_ALLOCATION", "45"))
OPTIONS_ALLOCATION: int = int(os.getenv("OPTIONS_ALLOCATION", "0"))
MOMO_ALLOCATION: int = int(os.getenv("MOMO_ALLOCATION", "25"))
CRYPTO_ALLOCATION: int = int(os.getenv("CRYPTO_ALLOCATION", "30"))

# ──────────────────────────────────────────────────────────────
# Risk Settings
# ──────────────────────────────────────────────────────────────
MAX_DAILY_RISK_PCT: float = float(os.getenv("MAX_DAILY_RISK_PCT", "8"))
MAX_SIMULTANEOUS_POSITIONS: int = int(os.getenv("MAX_SIMULTANEOUS_POSITIONS", "3"))
KILL_SWITCH_PCT: float = float(os.getenv("KILL_SWITCH_PCT", "12"))
INITIAL_CAPITAL: float = float(os.getenv("INITIAL_CAPITAL", "500"))

# ──────────────────────────────────────────────────────────────
# Trading Mode
# ──────────────────────────────────────────────────────────────
TRADING_MODE: str = os.getenv("TRADING_MODE", "paper")


# ──────────────────────────────────────────────────────────────
# Trading Phases
# ──────────────────────────────────────────────────────────────
@dataclass
class PhaseConfig:
    """Configuration for a single trading phase."""

    phase: int
    min_capital: float
    max_capital: float
    futures_contracts: int  # number of MNQ/NQ contracts
    futures_instrument: str  # "MNQ" or "NQ"
    futures_sl_pts: int
    futures_tp_pts: int
    options_max_capital: float
    momo_max_capital: float
    sessions: list[str] = field(default_factory=list)
    max_trades_per_day: int = 6


PHASES: dict[int, PhaseConfig] = {
    1: PhaseConfig(
        phase=1,
        min_capital=500.0,      # $500-$2,999 → MES (1 contract)
        max_capital=3000.0,
        futures_contracts=1,
        futures_instrument="MES",
        futures_sl_pts=10,
        futures_tp_pts=20,
        options_max_capital=50.0,
        momo_max_capital=100.0,
        sessions=["NY"],
        max_trades_per_day=4,
    ),
    2: PhaseConfig(
        phase=2,
        min_capital=3000.0,     # $3,000-$4,999 → MES (2 contracts)
        max_capital=5000.0,
        futures_contracts=2,
        futures_instrument="MES",
        futures_sl_pts=10,
        futures_tp_pts=20,
        options_max_capital=100.0,
        momo_max_capital=250.0,
        sessions=["London", "NY"],
        max_trades_per_day=6,
    ),
    3: PhaseConfig(
        phase=3,
        min_capital=5000.0,     # $5,000-$9,000 → MNQ (1 contract) AUTO-UPGRADE!
        max_capital=9000.0,
        futures_contracts=1,
        futures_instrument="MNQ",
        futures_sl_pts=10,
        futures_tp_pts=20,
        options_max_capital=150.0,
        momo_max_capital=600.0,
        sessions=["Tokyo", "London", "NY"],
        max_trades_per_day=8,
    ),
    4: PhaseConfig(
        phase=4,
        min_capital=9000.0,     # $9,000+ → NQ (1 contract)
        max_capital=15000.0,
        futures_contracts=1,
        futures_instrument="NQ",
        futures_sl_pts=12,
        futures_tp_pts=24,
        options_max_capital=300.0,
        momo_max_capital=1200.0,
        sessions=["Tokyo", "London", "NY"],
        max_trades_per_day=8,
    ),
}


# ──────────────────────────────────────────────────────────────
# Session Times (US Eastern Time, ET)
# ──────────────────────────────────────────────────────────────
@dataclass
class SessionTime:
    """Start and end hour (ET) for a trading session."""

    name: str
    start_hour: int
    start_minute: int
    end_hour: int
    end_minute: int


SESSIONS: dict[str, SessionTime] = {
    "Tokyo": SessionTime("Tokyo", 20, 0, 2, 0),   # 8:00 PM – 2:00 AM ET
    "London": SessionTime("London", 3, 0, 8, 0),   # 3:00 AM – 8:00 AM ET
    "NY": SessionTime("NY", 9, 30, 16, 0),         # 9:30 AM – 4:00 PM ET
    "PreMarket": SessionTime("PreMarket", 6, 0, 9, 25),  # 6:00 AM – 9:25 AM ET
    "Options": SessionTime("Options", 9, 30, 16, 15),   # 9:30 AM – 4:15 PM ET
    "Crypto_Asia": SessionTime("Crypto_Asia", 20, 0, 2, 0),
    "Crypto_Europe": SessionTime("Crypto_Europe", 3, 0, 8, 0),
    "Crypto_US": SessionTime("Crypto_US", 9, 30, 16, 0),
}


# ──────────────────────────────────────────────────────────────
# Risk Per Engine (daily % of total capital)
# ──────────────────────────────────────────────────────────────
ENGINE_DAILY_RISK_PCT: dict[str, float] = {
    "futures": 4.0,
    "options": 2.0,
    "momo": 2.0,
    "crypto": 2.0,
}


# ──────────────────────────────────────────────────────────────
# Capital Milestone Alerts
# ──────────────────────────────────────────────────────────────
@dataclass
class MilestoneAlert:
    """Capital milestone that triggers a withdrawal recommendation."""

    capital_threshold: float
    withdraw_amount: float
    message: str


MILESTONE_ALERTS: list[MilestoneAlert] = [
    MilestoneAlert(
        capital_threshold=2500.0,
        withdraw_amount=500.0,
        message="🎯 Milestone $2,500 reached! Consider withdrawing $500 (original capital).",
    ),
    MilestoneAlert(
        capital_threshold=7500.0,
        withdraw_amount=2000.0,
        message="🎯 Milestone $7,500 reached! Consider moving $2,000 to a safe account.",
    ),
    MilestoneAlert(
        capital_threshold=12000.0,
        withdraw_amount=3000.0,
        message="🎯 Milestone $12,000 reached! Consider moving $3,000 to a safe account.",
    ),
]


# ──────────────────────────────────────────────────────────────
# AI Brain Thresholds
# ──────────────────────────────────────────────────────────────
BRAIN_SCORE_FULL_SIZE: int = 75
BRAIN_SCORE_HALF_SIZE: int = 65

# ──────────────────────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────────────────────
TIMEZONE: str = "America/New_York"
BRAIN_MEMORY_FILE: str = "data/brain_memory.json"
RECONNECT_MAX_RETRIES: int = 5
RECONNECT_BASE_DELAY: float = 2.0  # seconds (exponential backoff base)
CONSECUTIVE_LOSS_PAUSE_HOURS: int = 4
MAX_CONSECUTIVE_LOSSES: int = 3
PDT_MAX_DAY_TRADES: int = 3
PDT_ROLLING_DAYS: int = 5
MOMO_EXECUTION_END_HOUR: int = int(os.getenv("MOMO_EXECUTION_END_HOUR", "12"))

# ──────────────────────────────────────────────────────────────
# AI Evaluator (Groq + Gemini)
# ──────────────────────────────────────────────────────────────
AI_EVALUATOR_ENABLED: bool = os.getenv("AI_EVALUATOR_ENABLED", "true").lower() == "true"
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
AI_EVALUATOR_TIMEOUT: float = float(os.getenv("AI_EVALUATOR_TIMEOUT", "3.0"))  # seconds

# ──────────────────────────────────────────────────────────────
# Adaptive Profit Protection
# ──────────────────────────────────────────────────────────────
# Profit tier thresholds — expressed as % of starting daily capital
# Tier 0 (normal): 0 % – PROFIT_TIER_1_PCT
# Tier 1:          PROFIT_TIER_1_PCT – PROFIT_TIER_2_PCT
# Tier 2:          PROFIT_TIER_2_PCT – PROFIT_TIER_3_PCT
# Tier 3:          PROFIT_TIER_3_PCT+
PROFIT_TIER_1_PCT: float = float(os.getenv("PROFIT_TIER_1_PCT", "10"))
PROFIT_TIER_2_PCT: float = float(os.getenv("PROFIT_TIER_2_PCT", "20"))
PROFIT_TIER_3_PCT: float = float(os.getenv("PROFIT_TIER_3_PCT", "30"))

# Minimum brain scores per tier (enforced on top of BRAIN_SCORE_HALF_SIZE)
PROFIT_TIER_0_MIN_SCORE: int = int(os.getenv("PROFIT_TIER_0_MIN_SCORE", "65"))
PROFIT_TIER_1_MIN_SCORE: int = int(os.getenv("PROFIT_TIER_1_MIN_SCORE", "75"))
PROFIT_TIER_2_MIN_SCORE: int = int(os.getenv("PROFIT_TIER_2_MIN_SCORE", "82"))
PROFIT_TIER_3_MIN_SCORE: int = int(os.getenv("PROFIT_TIER_3_MIN_SCORE", "90"))

# Position size multipliers per tier
PROFIT_TIER_0_SIZE_MULT: float = float(os.getenv("PROFIT_TIER_0_SIZE_MULT", "1.0"))
PROFIT_TIER_1_SIZE_MULT: float = float(os.getenv("PROFIT_TIER_1_SIZE_MULT", "0.75"))
PROFIT_TIER_2_SIZE_MULT: float = float(os.getenv("PROFIT_TIER_2_SIZE_MULT", "0.50"))
PROFIT_TIER_3_SIZE_MULT: float = float(os.getenv("PROFIT_TIER_3_SIZE_MULT", "0.25"))

# Trailing profit floor: once daily P&L exceeds tier-1 threshold (or the absolute
# USD threshold below, whichever is lower), the floor is set to
# max_pnl × PROFIT_FLOOR_RETENTION_PCT.
# If current P&L drops below this floor, no new trades are opened.
PROFIT_FLOOR_RETENTION_PCT: float = float(os.getenv("PROFIT_FLOOR_RETENTION_PCT", "0.70"))

# Absolute dollar gain that also activates the profit floor (independent of the
# percentage tier-1 threshold). Whichever threshold is hit first wins.
PROFIT_FLOOR_ACTIVATION_USD: float = float(os.getenv("PROFIT_FLOOR_ACTIVATION_USD", "25"))


def get_settings_summary() -> dict[str, Any]:
    """Return a summary of all active settings for logging/display."""
    return {
        "trading_mode": TRADING_MODE,
        "ibkr_host": IBKR_HOST,
        "ibkr_port": IBKR_PORT,
        "margin_account": IBKR_MARGIN_ACCOUNT or "(not set)",
        "cash_account": IBKR_CASH_ACCOUNT or "(not set)",
        "engines": {
            "futures": ENABLE_FUTURES,
            "options": ENABLE_OPTIONS,
            "momo": ENABLE_MOMO,
            "crypto": ENABLE_CRYPTO,
        },
        "allocation": {
            "futures": FUTURES_ALLOCATION,
            "options": OPTIONS_ALLOCATION,
            "momo": MOMO_ALLOCATION,
            "crypto": CRYPTO_ALLOCATION,
        },
        "risk": {
            "max_daily_pct": MAX_DAILY_RISK_PCT,
            "kill_switch_pct": KILL_SWITCH_PCT,
            "max_simultaneous_positions": MAX_SIMULTANEOUS_POSITIONS,
        },
        "initial_capital": INITIAL_CAPITAL,
    }
