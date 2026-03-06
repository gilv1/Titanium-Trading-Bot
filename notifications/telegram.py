"""
Telegram Notifier for Titanium Warrior v3.

Sends real-time trading notifications via python-telegram-bot (async).
All send methods are fire-and-forget with error handling.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from config import settings

if TYPE_CHECKING:
    from analysis.scanner import MomoCandidate
    from engines.base_engine import TradeResult

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Async Telegram notification service.

    Initialises the python-telegram-bot Bot instance lazily to avoid
    blocking at import time.
    """

    def __init__(self) -> None:
        self._bot: Any | None = None

    def _get_bot(self) -> Any | None:
        """Lazy-load the Bot instance."""
        if self._bot is not None:
            return self._bot
        if not settings.TELEGRAM_BOT_TOKEN:
            return None
        try:
            from telegram import Bot  # type: ignore

            self._bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
        except ImportError:
            logger.warning("python-telegram-bot not installed; Telegram disabled.")
        return self._bot

    async def _send(self, text: str) -> None:
        """Send a plain text message, ignoring all errors."""
        bot = self._get_bot()
        if bot is None or not settings.TELEGRAM_CHAT_ID:
            return
        try:
            await bot.send_message(
                chat_id=settings.TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="HTML",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram send failed: %s", exc)

    # ──────────────────────────────────────────────────────────
    # Trade notifications
    # ──────────────────────────────────────────────────────────

    async def send_trade_entry(self, trade_details: dict[str, Any]) -> None:
        """🟢 Trade entry notification."""
        engine = trade_details.get("engine", "").upper()
        ticker = trade_details.get("ticker", "")
        direction = trade_details.get("direction", "")
        entry = trade_details.get("entry", 0)
        sl = trade_details.get("sl", 0)
        tp = trade_details.get("tp", 0)
        qty = trade_details.get("qty", 0)
        score = trade_details.get("score", 0)
        rr = trade_details.get("rr", 0)

        direction_emoji = "🔼" if direction == "LONG" else "🔽"
        msg = (
            f"🟢 <b>TRADE ENTRY — {engine}</b>\n"
            f"{direction_emoji} <b>{ticker}</b> {direction}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📍 Entry:  <code>{entry:.4f}</code>\n"
            f"🛑 SL:     <code>{sl:.4f}</code>\n"
            f"🎯 TP:     <code>{tp:.4f}</code>\n"
            f"📦 Size:   <code>{qty}</code>\n"
            f"📊 R:R     <code>1:{rr}</code>\n"
            f"🧠 Score:  <code>{score}/100</code>"
        )
        await self._send(msg)

    async def send_trade_exit(self, trade_result: "TradeResult") -> None:
        """🔴/🟢 Trade exit notification."""
        won = trade_result.won
        emoji = "🟢 WIN" if won else "🔴 LOSS"
        pnl_sign = "+" if trade_result.pnl >= 0 else ""

        msg = (
            f"{emoji} — <b>{trade_result.engine.upper()}</b>\n"
            f"<b>{trade_result.ticker}</b> {trade_result.direction}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📍 Entry:  <code>{trade_result.entry_price:.4f}</code>\n"
            f"🏁 Exit:   <code>{trade_result.exit_price:.4f}</code>\n"
            f"💰 P&amp;L:    <code>{pnl_sign}{trade_result.pnl:.2f} ({pnl_sign}{trade_result.pnl_pct:.1f}%)</code>\n"
            f"⏱️ Duration: <code>{int(trade_result.duration_seconds)}s</code>\n"
            f"🧠 Score:  <code>{trade_result.ai_score}/100</code>"
        )
        await self._send(msg)

    # ──────────────────────────────────────────────────────────
    # Summary notifications
    # ──────────────────────────────────────────────────────────

    async def send_daily_summary(self, summary: dict[str, Any]) -> None:
        """📊 End-of-day summary."""
        pnl = summary.get("total_pnl", 0)
        pnl_sign = "+" if pnl >= 0 else ""
        msg = (
            f"📊 <b>DAILY SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 Total P&amp;L: <code>{pnl_sign}{pnl:.2f}</code>\n"
            f"📈 Win Rate:  <code>{summary.get('win_rate', 0):.0%}</code>\n"
            f"🔢 Trades:   <code>{summary.get('total_trades', 0)}</code>\n"
            f"💵 Capital:  <code>${summary.get('capital', 0):.2f}</code>\n"
            f"🏆 Phase:    <code>{summary.get('phase', 1)}</code>\n"
            f"\n<b>Per Engine:</b>\n"
        )
        for engine, stats in summary.get("engines", {}).items():
            e_pnl = stats.get("pnl", 0)
            e_sign = "+" if e_pnl >= 0 else ""
            msg += f"  • {engine.upper()}: <code>{e_sign}{e_pnl:.2f}</code> ({stats.get('trades', 0)} trades)\n"
        await self._send(msg)

    async def send_phase_change(self, old_phase: int, new_phase: int) -> None:
        """🚀 Phase upgrade celebration."""
        msg = (
            f"🚀 <b>PHASE UPGRADE!</b>\n"
            f"Phase <code>{old_phase}</code> → <code>{new_phase}</code>\n"
            f"Scaling up position sizes. Let's go! 💪"
        )
        await self._send(msg)

    async def send_kill_switch(self, reason: str) -> None:
        """🚨 Emergency kill switch notification."""
        msg = (
            f"🚨 <b>KILL SWITCH ACTIVATED</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⛔ Reason: <code>{reason}</code>\n"
            f"All engines halted for <code>24h</code>.\n"
            f"Review your risk settings before resuming."
        )
        await self._send(msg)

    async def send_momo_scanner(self, candidates: list["MomoCandidate"]) -> None:
        """📡 Pre-market MoMo scanner results."""
        if not candidates:
            await self._send("📡 <b>MOMO SCANNER</b>\nNo candidates passed filters today.")
            return

        lines = ["📡 <b>PRE-MARKET MOMO SCANNER</b>", "━━━━━━━━━━━━━━━━"]
        for i, c in enumerate(candidates[:5], 1):
            fire = "🔥" if c.score > 80 else ("⚡" if c.score >= 65 else "📌")
            lines.append(
                f"{fire} <b>{c.ticker}</b> — Score: <code>{c.score}/110</code>\n"
                f"   Gap: <code>+{c.gap_pct:.1f}%</code> | RVOL: <code>{c.rvol:.1f}×</code> "
                f"| Float: <code>{c.float_shares:.1f}M</code> | Price: <code>${c.price:.2f}</code>\n"
                f"   📰 {c.news_headline[:60]}…" if c.news_headline else ""
            )
        await self._send("\n".join(lines))

    async def send_milestone_alert(self, milestone_msg: str) -> None:
        """💰 Capital milestone withdrawal recommendation."""
        await self._send(f"💰 <b>CAPITAL MILESTONE</b>\n{milestone_msg}")

    async def send_profit_tier_alert(
        self,
        tier: int,
        pnl: float,
        min_score: int,
        size_multiplier: float,
    ) -> None:
        """🛡️ Profit protection tier change alert."""
        tier_emoji = ["🟢", "🟡", "🟠", "🔴"][min(tier, 3)]
        pnl_sign = "+" if pnl >= 0 else ""
        msg = (
            f"{tier_emoji} <b>PROFIT PROTECTION — TIER {tier}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 Daily P&amp;L:  <code>{pnl_sign}{pnl:.2f}</code>\n"
            f"🧠 Min Score:  <code>{min_score}/100</code>\n"
            f"📦 Size Mult:  <code>{size_multiplier:.0%}</code>\n"
            f"⚠️ Only high-conviction trades will be approved."
        )
        await self._send(msg)

    async def send_profit_floor_alert(
        self,
        activated: bool,
        pnl: float,
        floor_value: float,
    ) -> None:
        """🛡️ Trailing profit floor alert."""
        pnl_sign = "+" if pnl >= 0 else ""
        if activated:
            msg = (
                f"🛡️ <b>PROFIT FLOOR ACTIVATED</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💰 Peak P&amp;L:   <code>{pnl_sign}{pnl:.2f}</code>\n"
                f"🔒 Floor set at: <code>+{floor_value:.2f}</code>\n"
                f"📉 If P&amp;L drops to floor → no new trades."
            )
        else:
            msg = (
                f"🚫 <b>PROFIT FLOOR HIT — TRADING PAUSED</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💰 Current P&amp;L: <code>{pnl_sign}{pnl:.2f}</code>\n"
                f"🔒 Floor was:    <code>+{floor_value:.2f}</code>\n"
                f"⛔ No new trades until tomorrow. Existing SL/TP still active."
            )
        await self._send(msg)

    async def send_weekly_summary(self, summary: dict[str, Any]) -> None:
        """📈 Weekly performance review."""
        pnl = summary.get("total_pnl", 0)
        pnl_sign = "+" if pnl >= 0 else ""
        msg = (
            f"📈 <b>WEEKLY SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 Total P&amp;L: <code>{pnl_sign}{pnl:.2f}</code>\n"
            f"📈 Win Rate:  <code>{summary.get('win_rate', 0):.0%}</code>\n"
            f"🔢 Trades:   <code>{summary.get('total_trades', 0)}</code>\n"
            f"📉 Max DD:   <code>{summary.get('max_drawdown', 0):.2f}</code>\n"
            f"💵 Capital:  <code>${summary.get('capital', 0):.2f}</code>\n"
            f"🏆 Phase:    <code>{summary.get('phase', 1)}</code>"
        )
        await self._send(msg)
