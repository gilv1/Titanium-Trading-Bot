"""
IBKR Connection Manager for Titanium Warrior v3.

Manages two simultaneous ib_insync connections:
  - Margin account (futures + crypto) — clientId=1
  - Cash account (options + momo stocks) — clientId=2

Features:
  - Auto-reconnect with exponential backoff (max 5 retries)
  - Connection health monitoring
  - Async-compatible using ib_insync's asyncio integration
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


class IBKRConnection:
    """
    Manages a single IBKR connection via ib_insync.

    Wraps the IB object with auto-reconnect logic and exposes
    convenience helpers for account value queries and order placement.
    """

    def __init__(self, account: str, client_id: int, label: str) -> None:
        self.account = account
        self.client_id = client_id
        self.label = label
        self._ib: Any | None = None  # ib_insync.IB instance (imported lazily)
        self._connected = False

    # ──────────────────────────────────────────────────────────
    # Connection lifecycle
    # ──────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway with exponential backoff retry."""
        # Lazy import so the rest of the module loads without ib_insync
        try:
            from ib_insync import IB  # type: ignore
        except ImportError:
            logger.error("ib_insync is not installed. Run: pip install ib_insync")
            return False

        delay = settings.RECONNECT_BASE_DELAY
        for attempt in range(1, settings.RECONNECT_MAX_RETRIES + 1):
            try:
                self._ib = IB()
                await self._ib.connectAsync(
                    host=settings.IBKR_HOST,
                    port=settings.IBKR_PORT,
                    clientId=self.client_id,
                    account=self.account,
                )
                self._connected = True
                logger.info(
                    "[%s] Connected to IBKR (account=%s, clientId=%d)",
                    self.label,
                    self.account,
                    self.client_id,
                )
                # Register disconnect callback — remove first to avoid double-registration
                try:
                    self._ib.disconnectedEvent -= self._on_disconnected
                except Exception:  # noqa: BLE001
                    pass
                self._ib.disconnectedEvent += self._on_disconnected
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] Connection attempt %d/%d failed: %s. Retrying in %.1fs…",
                    self.label,
                    attempt,
                    settings.RECONNECT_MAX_RETRIES,
                    exc,
                    delay,
                )
                if attempt < settings.RECONNECT_MAX_RETRIES:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60.0)

        logger.error("[%s] All %d connection attempts failed.", self.label, settings.RECONNECT_MAX_RETRIES)
        return False

    def _on_disconnected(self) -> None:
        """Called by ib_insync when connection is lost; schedule reconnect."""
        self._connected = False
        logger.warning("[%s] Disconnected! Scheduling reconnect...", self.label)
        loop = asyncio.get_event_loop()
        loop.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Reconnect with exponential backoff."""
        for attempt in range(1, settings.RECONNECT_MAX_RETRIES + 1):
            delay = min(
                settings.RECONNECT_BASE_DELAY * (2 ** (attempt - 1)),
                60.0,  # cap at 60 seconds
            )
            logger.info(
                "[%s] Reconnect attempt %d/%d in %.1fs...",
                self.label,
                attempt,
                settings.RECONNECT_MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)
            success = await self.connect()
            if success:
                logger.info("[%s] Reconnected successfully!", self.label)
                return
        logger.error("[%s] Failed to reconnect after %d attempts.", self.label, settings.RECONNECT_MAX_RETRIES)

    async def disconnect(self) -> None:
        """Gracefully disconnect."""
        if self._ib is not None and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("[%s] Disconnected.", self.label)

    def is_connected(self) -> bool:
        """Return True if the connection is live."""
        if self._ib is None:
            return False
        return self._connected and self._ib.isConnected()

    # ──────────────────────────────────────────────────────────
    # Account & order helpers
    # ──────────────────────────────────────────────────────────

    def get_account_value(self, tag: str = "NetLiquidation") -> float:
        """Return a numeric account value (e.g. net liquidation)."""
        if not self.is_connected():
            logger.warning("[%s] Not connected; cannot fetch account value.", self.label)
            return 0.0
        try:
            values = self._ib.accountValues(account=self.account)
            for av in values:
                if av.tag == tag and av.currency == "USD":
                    return float(av.value)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Error fetching account value: %s", self.label, exc)
        return 0.0

    async def place_order(self, contract: Any, order: Any) -> Any | None:
        """Place an order and return the Trade object, or None on error."""
        if not self.is_connected():
            logger.error("[%s] Cannot place order: not connected.", self.label)
            return None
        try:
            order.account = self.account  # required when multiple accounts share one login
            trade = self._ib.placeOrder(contract, order)
            await asyncio.sleep(0)  # yield to allow IB event loop to process
            logger.info(
                "[%s] Order placed: %s %s qty=%s @ %s",
                self.label,
                order.action,
                contract.symbol,
                order.totalQuantity,
                getattr(order, "lmtPrice", "MKT"),
            )
            return trade
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Order placement error: %s", self.label, exc)
            return None

    async def cancel_order(self, trade: Any) -> None:
        """Cancel an open order."""
        if not self.is_connected() or trade is None:
            return
        try:
            self._ib.cancelOrder(trade.order)
            logger.info("[%s] Order cancelled: orderId=%s", self.label, trade.order.orderId)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Cancel order error: %s", self.label, exc)

    def get_ib(self) -> Any | None:
        """Return the raw IB instance for advanced use."""
        return self._ib


class ConnectionManager:
    """
    Top-level connection manager that owns both IBKR connections.

    Usage::

        mgr = ConnectionManager()
        await mgr.connect_margin()
        await mgr.connect_cash()
    """

    def __init__(self) -> None:
        self.margin = IBKRConnection(
            account=settings.IBKR_MARGIN_ACCOUNT,
            client_id=1,
            label="MARGIN",
        )
        self.cash = IBKRConnection(
            account=settings.IBKR_CASH_ACCOUNT,
            client_id=2,
            label="CASH",
        )

    async def connect_margin(self) -> bool:
        """Connect the margin account (futures + crypto)."""
        return await self.margin.connect()

    async def connect_cash(self) -> bool:
        """Connect the cash account (options + momo)."""
        return await self.cash.connect()

    async def disconnect(self) -> None:
        """Disconnect both accounts."""
        await self.margin.disconnect()
        await self.cash.disconnect()

    def is_connected(self) -> dict[str, bool]:
        """Return connection status for both accounts."""
        return {
            "margin": self.margin.is_connected(),
            "cash": self.cash.is_connected(),
        }
