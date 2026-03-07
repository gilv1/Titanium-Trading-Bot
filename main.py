"""
Titanium Warrior v3 — Main Entry Point.

Usage:
    python main.py             # paper trading (default)
    python main.py --paper     # paper trading
    python main.py --live      # live trading
    python main.py --backtest  # run backtest

Initialises all components and starts enabled engines as async tasks.
Handles graceful shutdown on SIGINT / SIGTERM.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from typing import Any

# ──────────────────────────────────────────────────────────────
# ASCII Banner
# ──────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ████████╗██╗████████╗ █████╗ ███╗   ██╗██╗██╗   ██╗███╗  ║
║      ██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║██║██║   ██║████╗ ║
║      ██║   ██║   ██║   ███████║██╔██╗ ██║██║██║   ██║██╔██╗║
║      ██║   ██║   ██║   ██╔══██║██║╚██╗██║██║██║   ██║██║╚██║
║      ██║   ██║   ██║   ██║  ██║██║ ╚████║██║╚██████╔╝██║ ╚█║
║      ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝ ╚═════╝ ╚═╝  ║
║                                                              ║
║              W A R R I O R   v 3                             ║
║                                                              ║
║   🔵 Motor 1: FUTURES (MNQ/NQ)    — Margin Account          ║
║   🟢 Motor 2: OPTIONS 0DTE        — Cash Account (off)      ║
║   🟡 Motor 3: MOMO Small-Caps     — Cash Account (off)      ║
║   🟠 Motor 4: CRYPTO (BTC/ETH)    — Margin Account          ║
║                                                              ║
║   🧠 AI Brain  — AUTOEVOLUTIVE self-learning engine          ║
║   💰 Reto Tracker — $500 → $15,000 compound auto-scale      ║
║   📱 Telegram — Real-time trade notifications                ║
╚══════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("titanium")


# ──────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Titanium Warrior v3 Trading Bot")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--paper", action="store_true", default=True, help="Run in paper-trading mode (default)")
    group.add_argument("--live", action="store_true", help="Run in live-trading mode")
    group.add_argument("--backtest", action="store_true", help="Run backtest")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Engine factory
# ──────────────────────────────────────────────────────────────


def build_engines(
    connection: Any,
    brain: Any,
    reto: Any,
    risk: Any,
    telegram: Any,
    journal: Any = None,
) -> list[Any]:
    """Instantiate enabled engines and return them as a list."""
    from config import settings

    engines: list[Any] = []

    if settings.ENABLE_FUTURES:
        from engines.futures_engine import FuturesEngine
        engines.append(FuturesEngine(connection, brain, reto, risk, telegram, journal=journal))
        logger.info("🔵 Futures engine ENABLED")
    else:
        logger.info("🔵 Futures engine disabled")

    if settings.ENABLE_OPTIONS:
        from engines.options_engine import OptionsEngine
        engines.append(OptionsEngine(connection, brain, reto, risk, telegram, journal=journal))
        logger.info("🟢 Options engine ENABLED")
    else:
        logger.info("🟢 Options engine disabled")

    if settings.ENABLE_MOMO:
        from engines.momo_engine import MomoEngine
        engines.append(MomoEngine(connection, brain, reto, risk, telegram, journal=journal))
        logger.info("🟡 MoMo engine ENABLED")
    else:
        logger.info("🟡 MoMo engine disabled")

    if settings.ENABLE_CRYPTO:
        from engines.crypto_engine import CryptoEngine
        engines.append(CryptoEngine(connection, brain, reto, risk, telegram, journal=journal))
        logger.info("🟠 Crypto engine ENABLED")
    else:
        logger.info("🟠 Crypto engine disabled")

    return engines


# ──────────────────────────────────────────────────────────────
# Graceful shutdown
# ──────────────────────────────────────────────────────────────

_shutdown_event = asyncio.Event()


def _handle_signal(*_: Any) -> None:
    logger.info("Shutdown signal received.")
    _shutdown_event.set()


# ──────────────────────────────────────────────────────────────
# Main async runner
# ──────────────────────────────────────────────────────────────


async def run_bot(trading_mode: str) -> None:
    """Initialise all components and start enabled engine tasks."""
    from config import settings
    from config.settings import get_settings_summary
    from core.brain import AIBrain
    from core.connection import ConnectionManager
    from core.reto_tracker import RetoTracker
    from core.risk_manager import RiskManager
    from journal.trade_journal import TradeJournal
    from notifications.telegram import TelegramNotifier

    # ── 1. Settings ─────────────────────────────────────────
    os.environ.setdefault("TRADING_MODE", trading_mode)
    summary = get_settings_summary()
    logger.info("Settings loaded: mode=%s capital=$%.0f", trading_mode, settings.INITIAL_CAPITAL)

    # ── 2. Components ────────────────────────────────────────
    connection = ConnectionManager()
    brain = AIBrain()
    reto = RetoTracker(initial_capital=settings.INITIAL_CAPITAL)
    risk = RiskManager(reto_tracker=reto)
    telegram = TelegramNotifier()
    journal = TradeJournal()

    logger.info(
        "Startup — Phase %d | Capital $%.2f | Instrument %s",
        reto.get_phase(),
        reto.capital,
        reto.get_futures_instrument(),
    )

    # ── 3. Connections ───────────────────────────────────────
    margin_ok = await connection.connect_margin()
    cash_ok = await connection.connect_cash()
    if not margin_ok:
        logger.warning("Margin account not connected — futures/crypto engines will be inactive.")
    if not cash_ok:
        logger.warning("Cash account not connected — options/momo engines will be inactive.")

    # ── 4. Engines ───────────────────────────────────────────
    engines = build_engines(connection, brain, reto, risk, telegram, journal=journal)
    if not engines:
        logger.warning("No engines enabled. Check your .env settings.")
        return

    # ── 5. Startup notification ──────────────────────────────
    engine_names = [e.get_engine_name() for e in engines]
    await telegram._send(
        f"🚀 <b>Titanium Warrior v3 started</b>\n"
        f"Mode: <code>{trading_mode}</code>\n"
        f"Engines: <code>{', '.join(engine_names)}</code>\n"
        f"Phase: <code>{reto.get_phase()}</code> | Capital: <code>${reto.capital:.2f}</code>"
    )

    # ── 6. Run engines as concurrent tasks ───────────────────
    tasks = [asyncio.create_task(engine.start(), name=engine.get_engine_name()) for engine in engines]

    # Register OS signals for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    logger.info("All engines running. Press Ctrl+C to stop.")
    await _shutdown_event.wait()

    # ── 7. Graceful shutdown ─────────────────────────────────
    logger.info("Shutting down engines…")
    for engine in engines:
        await engine.stop()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await connection.disconnect()
    brain.save_memory()

    # Daily summary
    summary_data = journal.get_daily_summary()
    await telegram.send_daily_summary(
        {
            "total_pnl": summary_data.total_pnl,
            "win_rate": summary_data.win_rate,
            "total_trades": summary_data.total_trades,
            "capital": reto.capital,
            "phase": reto.get_phase(),
            "engines": summary_data.engines,
        }
    )
    logger.info("Titanium Warrior v3 stopped cleanly.")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────


def main() -> None:
    print(BANNER)

    args = parse_args()
    if args.backtest:
        logger.info("Launching backtest…")
        import backtest
        backtest.run()
        return

    mode = "live" if args.live else "paper"
    logger.info("Starting Titanium Warrior v3 in %s mode…", mode)
    asyncio.run(run_bot(trading_mode=mode))


if __name__ == "__main__":
    main()
