# engine/order.py
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, getcontext
from uuid import uuid4
import time
from typing import Optional

# Set high precision for financial calculations
getcontext().prec = 28


@dataclass
class Order:
    """
    Enhanced Order dataclass with validation and better error handling.

    Fields:
        symbol: trading pair (e.g., "BTC-USDT")
        side: "buy" or "sell"
        quantity: Decimal (initial quantity)
        price: Decimal | None (None for market orders)
        order_type: "market", "limit", "ioc", "fok"
        order_id: unique id string
        timestamp: float (epoch seconds)
        remaining: Decimal (remaining qty to fill)
        filled: Decimal (filled quantity)
        status: order status string
    """
    symbol: str
    side: str                    # "buy" or "sell"
    quantity: Decimal
    price: Optional[Decimal]     # None for market orders
    order_type: str              # "market", "limit", "ioc", "fok"
    order_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    remaining: Optional[Decimal] = None
    filled: Decimal = field(default_factory=lambda: Decimal("0"))
    status: str = field(default="pending")
    client_order_id: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize order data."""
        self._validate_and_normalize()
        if self.remaining is None:
            self.remaining = Decimal(self.quantity)

    def _validate_and_normalize(self):
        """Basic validation and type conversion."""
        # Normalize strings
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("Symbol must be a non-empty string")
        self.symbol = self.symbol.upper().strip()

        # Validate side
        if isinstance(self.side, str):
            self.side = self.side.lower().strip()
        if self.side not in ("buy", "sell"):
            raise ValueError("Side must be 'buy' or 'sell'")

        # Validate order type
        if isinstance(self.order_type, str):
            self.order_type = self.order_type.lower().strip()
        if self.order_type not in ("market", "limit", "ioc", "fok"):
            raise ValueError("Invalid order type")

        # Convert and validate quantity
        try:
            if isinstance(self.quantity, (int, float, str)):
                self.quantity = Decimal(str(self.quantity))
            if self.quantity <= 0:
                raise ValueError("Quantity must be positive")
        except (InvalidOperation, TypeError) as e:
            raise ValueError(f"Invalid quantity: {e}")

        # Convert and validate price
        if self.price is not None:
            try:
                if isinstance(self.price, (int, float, str)):
                    self.price = Decimal(str(self.price))
                if self.price <= 0:
                    raise ValueError("Price must be positive")
            except (InvalidOperation, TypeError) as e:
                raise ValueError(f"Invalid price: {e}")

        # Validate price requirements
        if self.order_type in ("limit", "ioc", "fok") and self.price is None:
            raise ValueError(f"{self.order_type} orders require a price")
        
        if self.order_type == "market" and self.price is not None:
            raise ValueError("Market orders cannot have a price")

        # Convert remaining and filled if provided
        if self.remaining is not None:
            try:
                if isinstance(self.remaining, (int, float, str)):
                    self.remaining = Decimal(str(self.remaining))
            except (InvalidOperation, TypeError) as e:
                raise ValueError(f"Invalid remaining quantity: {e}")

        if isinstance(self.filled, (int, float, str)):
            self.filled = Decimal(str(self.filled))

    @property
    def is_active(self) -> bool:
        """Check if order is active (can be matched or cancelled)."""
        return self.status in ("pending", "partially_filled")

    def is_market(self) -> bool:
        """Check if this is a market order."""
        return self.order_type == "market"

    def is_limit(self) -> bool:
        """Check if this is a limit order."""
        return self.order_type == "limit"

    def is_ioc(self) -> bool:
        """Check if this is an IOC order."""
        return self.order_type == "ioc"

    def is_fok(self) -> bool:
        """Check if this is a FOK order."""
        return self.order_type == "fok"

    def fill(self, quantity: Decimal) -> None:
        """
        Fill part of the order.
        
        Args:
            quantity: Amount to fill
        """
        if quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        
        if quantity > self.remaining:
            raise ValueError("Cannot fill more than remaining quantity")
        
        self.filled += quantity
        self.remaining -= quantity
        
        # Update status
        if self.remaining <= 0:
            self.status = "filled"
        elif self.filled > 0:
            self.status = "partially_filled"

    def cancel(self) -> bool:
        """Cancel the order if it's active."""
        if not self.is_active:
            return False
        
        self.status = "cancelled"
        return True

    def reject(self, reason: str = None) -> None:
        """Reject the order."""
        self.status = "rejected"

    def to_dict(self) -> dict:
        """Convert order to dictionary for serialization."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": str(self.quantity),
            "price": str(self.price) if self.price is not None else None,
            "remaining": str(self.remaining),
            "filled": str(self.filled),
            "status": self.status,
            "timestamp": self.timestamp
        }

    def __repr__(self) -> str:
        price_str = str(self.price) if self.price is not None else "MKT"
        return (f"<Order {self.order_id[:8]} {self.side.upper()} "
                f"{self.quantity}@{price_str} ({self.order_type}) {self.status}>")

    def __eq__(self, other) -> bool:
        """Orders are equal if they have the same order_id."""
        if not isinstance(other, Order):
            return False
        return self.order_id == other.order_id

    def __hash__(self) -> int:
        """Hash based on order_id for use in sets/dicts."""
        return hash(self.order_id)


# Convenience factory functions for common order types
def create_market_order(symbol: str, side: str, quantity, client_order_id: str = None) -> Order:
    """Create a market order."""
    return Order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=None,
        order_type="market",
        client_order_id=client_order_id
    )

def create_limit_order(symbol: str, side: str, quantity, price, client_order_id: str = None) -> Order:
    """Create a limit order."""
    return Order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        order_type="limit",
        client_order_id=client_order_id
    )






