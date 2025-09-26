# engine/book.py
from collections import deque
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple, Set
from engine.order import Order
import uuid
import time
import threading
import logging
from sortedcontainers import SortedDict
import bisect

# Set Decimal precision for financial calculations
getcontext().prec = 28

logger = logging.getLogger(__name__)


class PriceLevel:
    """
    Enhanced price level with better order management and thread safety.
    """
    def __init__(self, price: Decimal):
        self.price: Decimal = price
        self.orders: deque[Order] = deque()
        self.aggregate: Decimal = Decimal("0")
        self.order_map: Dict[str, Order] = {}  # For O(1) order lookup
        self._lock = threading.RLock()

    def add_order(self, order: Order) -> None:
        """Thread-safe order addition."""
        with self._lock:
            self.orders.append(order)
            self.order_map[order.order_id] = order
            self.aggregate += order.remaining

    def peek_oldest(self) -> Optional[Order]:
        """Get the oldest order without removing it."""
        with self._lock:
            return self.orders[0] if self.orders else None

    def pop_oldest(self) -> Optional[Order]:
        """Remove and return the oldest order."""
        with self._lock:
            if not self.orders:
                return None
            order = self.orders.popleft()
            self.order_map.pop(order.order_id, None)
            self.aggregate -= order.remaining
            return order

    def remove_order(self, order_id: str) -> bool:
        """
        Remove a specific order by ID with O(1) lookup.
        Returns True if removed, False if not found.
        """
        with self._lock:
            if order_id not in self.order_map:
                return False
            
            order_to_remove = self.order_map.pop(order_id)
            
            # Remove from deque (this is O(n) but unavoidable)
            new_orders = deque()
            for order in self.orders:
                if order.order_id != order_id:
                    new_orders.append(order)
            
            self.orders = new_orders
            self.aggregate -= order_to_remove.remaining
            return True

    def update_order_fill(self, order_id: str, filled_qty: Decimal) -> bool:
        """Update aggregate when an order is partially filled."""
        with self._lock:
            if order_id in self.order_map:
                self.aggregate -= filled_qty
                return True
            return False

    def is_empty(self) -> bool:
        """Check if price level is empty."""
        with self._lock:
            return len(self.orders) == 0 or self.aggregate <= 0

    def __len__(self) -> int:
        return len(self.orders)


class Trade:
    """Enhanced trade representation."""
    def __init__(self, symbol: str, price: Decimal, quantity: Decimal, 
                 maker_order: Order, taker_order: Order, trade_seq: int):
        self.trade_id = f"{symbol}-{trade_seq}-{str(uuid.uuid4())[:8]}"
        self.timestamp = time.time()
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.maker_order_id = maker_order.order_id
        self.taker_order_id = taker_order.order_id
        self.aggressor_side = taker_order.side
        self.trade_seq = trade_seq

    def to_dict(self) -> dict:
        """Convert trade to dictionary for API response."""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "trade_id": self.trade_id,
            "price": str(self.price),
            "quantity": str(self.quantity),
            "aggressor_side": self.aggressor_side,
            "maker_order_id": self.maker_order_id,
            "taker_order_id": self.taker_order_id
        }


class OrderBook:
    """
    Enhanced OrderBook with thread safety, better performance, and comprehensive error handling.
    """
    def __init__(self, symbol: str, tick_size: Decimal = Decimal("0.01"), 
                 min_quantity: Decimal = Decimal("0.00000001")):
        self.symbol: str = symbol
        self.tick_size: Decimal = tick_size
        self.min_quantity: Decimal = min_quantity
        
        # Use SortedDict for O(log n) insertions and efficient range queries
        # Keys are negative for bids to maintain descending order
        self.bids: SortedDict[Decimal, PriceLevel] = SortedDict()
        self.asks: SortedDict[Decimal, PriceLevel] = SortedDict()
        
        # Order tracking
        self.orders: Dict[str, Tuple[Order, PriceLevel]] = {}
        self.client_order_map: Dict[str, str] = {}  # client_order_id -> order_id
        
        # Trade tracking
        self.trade_seq: int = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "total_orders": 0,
            "total_trades": 0,
            "total_volume": Decimal("0"),
            "last_trade_price": None,
            "last_trade_time": None
        }

    def _validate_price(self, price: Decimal) -> Decimal:
        """Validate and normalize price to tick size."""
        if price <= 0:
            raise ValueError("Price must be positive")
        
        # Round to nearest tick
        ticks = price / self.tick_size
        rounded_ticks = round(ticks)
        return rounded_ticks * self.tick_size

    def _validate_quantity(self, quantity: Decimal) -> None:
        """Validate quantity."""
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if quantity < self.min_quantity:
            raise ValueError(f"Quantity below minimum: {self.min_quantity}")

    def best_bid(self) -> Optional[Decimal]:
        """Get best bid price (highest)."""
        with self._lock:
            return -self.bids.peekitem(-1)[0] if self.bids else None

    def best_ask(self) -> Optional[Decimal]:
        """Get best ask price (lowest)."""
        with self._lock:
            return self.asks.peekitem(0)[0] if self.asks else None

    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        bb, ba = self.best_bid(), self.best_ask()
        return ba - bb if bb and ba else None

    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price."""
        bb, ba = self.best_bid(), self.best_ask()
        return (bb + ba) / 2 if bb and ba else None

    def get_bbo(self) -> Dict:
        """Get Best Bid Offer snapshot."""
        with self._lock:
            bb = self.best_bid()
            ba = self.best_ask()
            return {
                "symbol": self.symbol,
                "best_bid": str(bb) if bb else None,
                "best_ask": str(ba) if ba else None,
                "spread": str(self.spread()) if self.spread() else None,
                "mid_price": str(self.mid_price()) if self.mid_price() else None,
                "timestamp": time.time()
            }

    def get_depth(self, levels: int = 10) -> Dict:
        """Get order book depth (L2 data)."""
        with self._lock:
            bids = []
            asks = []
            
            # Get top N bid levels (highest prices first)
            # Need to reverse the items manually since SortedDict doesn't support reverse parameter
            bid_items = list(self.bids.items())
            bid_items.reverse()  # Reverse to get highest prices first
            
            for neg_price, level in bid_items:
                if len(bids) >= levels:
                    break
                if level.aggregate > 0:  # Only show levels with quantity
                    price = -neg_price
                    bids.append([str(price), str(level.aggregate)])
            
            # Get top N ask levels (lowest prices first)
            for price, level in self.asks.items():
                if len(asks) >= levels:
                    break
                if level.aggregate > 0:  # Only show levels with quantity
                    asks.append([str(price), str(level.aggregate)])
            
            return {
                "timestamp": time.time(),
                "symbol": self.symbol,
                "bids": bids,
                "asks": asks
            }

    def _add_resting_order(self, order: Order) -> None:
        """Add a resting limit order to the book."""
        price = self._validate_price(order.price)
        
        if order.side == "buy":
            neg_price = -price  # Store as negative for descending sort
            if neg_price not in self.bids:
                self.bids[neg_price] = PriceLevel(price)
            level = self.bids[neg_price]
        else:
            if price not in self.asks:
                self.asks[price] = PriceLevel(price)
            level = self.asks[price]
        
        level.add_order(order)
        self.orders[order.order_id] = (order, level)
        
        if order.client_order_id:
            self.client_order_map[order.client_order_id] = order.order_id

    def _remove_empty_level(self, price: Decimal, side: str) -> None:
        """Remove empty price level."""
        if side == "buy":
            neg_price = -price
            if neg_price in self.bids and self.bids[neg_price].is_empty():
                del self.bids[neg_price]
        else:
            if price in self.asks and self.asks[price].is_empty():
                del self.asks[price]

    def cancel_order(self, order_id: str = None, client_order_id: str = None) -> bool:
        """
        Cancel order by order_id or client_order_id.
        Returns True if cancelled, False if not found.
        """
        with self._lock:
            # Resolve client_order_id to order_id
            if client_order_id and not order_id:
                order_id = self.client_order_map.get(client_order_id)
            
            if not order_id or order_id not in self.orders:
                return False
            
            order, level = self.orders.pop(order_id)
            
            # Remove from client mapping if exists
            if order.client_order_id:
                self.client_order_map.pop(order.client_order_id, None)
            
            # Cancel the order
            order.cancel()
            
            # Remove from price level
            success = level.remove_order(order_id)
            
            # Clean up empty level
            if level.is_empty():
                self._remove_empty_level(level.price, order.side)
            
            return success

    def _calculate_available_liquidity(self, incoming_side: str, max_price: Decimal = None) -> Decimal:
        """
        Calculate total available liquidity for matching.
        Improved FOK calculation that considers all marketable levels.
        """
        total = Decimal("0")
        
        if incoming_side == "buy":
            # Buyer takes asks from lowest price upward
            for price, level in self.asks.items():
                if max_price and price > max_price:
                    break
                total += level.aggregate
        else:
            # Seller takes bids from highest price downward
            for neg_price, level in self.bids.items(reverse=True):
                price = -neg_price
                if max_price and price < max_price:
                    break
                total += level.aggregate
        
        return total

    def _is_order_marketable(self, order: Order) -> bool:
        """Check if order can be immediately matched."""
        if order.is_market():
            return True
        
        if not order.price:
            return False
            
        if order.side == "buy":
            best_ask = self.best_ask()
            return best_ask is not None and order.price >= best_ask
        else:
            best_bid = self.best_bid()
            return best_bid is not None and order.price <= best_bid

    def _execute_trade(self, maker_order: Order, taker_order: Order, 
                      quantity: Decimal, price: Decimal) -> Trade:
        """Execute a trade between two orders."""
        # Update order quantities
        maker_order.fill(quantity)
        taker_order.fill(quantity)
        
        # Update price level aggregate
        if maker_order.order_id in self.orders:
            _, level = self.orders[maker_order.order_id]
            level.update_order_fill(maker_order.order_id, quantity)
        
        # Create trade record
        self.trade_seq += 1
        trade = Trade(
            symbol=self.symbol,
            price=price,
            quantity=quantity,
            maker_order=maker_order,
            taker_order=taker_order,
            trade_seq=self.trade_seq
        )
        
        # Update statistics
        self.stats["total_trades"] += 1
        self.stats["total_volume"] += quantity
        self.stats["last_trade_price"] = price
        self.stats["last_trade_time"] = trade.timestamp
        
        return trade

    def _match_order(self, incoming_order: Order) -> List[Trade]:
        """
        Core matching logic with improved error handling and FOK support.
        """
        trades: List[Trade] = []
        
        # Choose opposite side of book
        if incoming_order.side == "buy":
            opposite_levels = self.asks
            price_check = lambda p: incoming_order.is_market() or incoming_order.price >= p
        else:
            opposite_levels = self.bids
            price_check = lambda p: incoming_order.is_market() or incoming_order.price <= (-p if incoming_order.side == "sell" else p)
        
        # FOK pre-validation: ensure sufficient liquidity
        if incoming_order.is_fok():
            max_price = incoming_order.price if not incoming_order.is_market() else None
            available = self._calculate_available_liquidity(incoming_order.side, max_price)
            if available < incoming_order.remaining:
                logger.info(f"FOK order {incoming_order.order_id} rejected: insufficient liquidity")
                incoming_order.reject("Insufficient liquidity")
                return []
        
        # Process opposite levels in price-time priority
        levels_to_remove = []
        
        if incoming_order.side == "buy":
            # Take asks from lowest to highest (natural order)
            level_items = list(opposite_levels.items())
        else:
            # Take bids from highest to lowest (reverse order)
            level_items = list(opposite_levels.items())
            level_items.reverse()
        
        for key, level in level_items:
            if incoming_order.remaining <= 0:
                break
            
            # Get the actual price for this level
            if incoming_order.side == "buy":
                level_price = key  # For asks, key is the actual price
            else:
                level_price = -key  # For bids, key is negative, so convert back
            
            # Check if this price level is marketable
            if incoming_order.side == "buy":
                # For buy orders, check against ask price
                if not price_check(level_price):
                    break
            else:
                # For sell orders, check against bid price (key is negative)
                if not price_check(key):
                    break
            
            # Match against orders at this price level
            while incoming_order.remaining > 0 and not level.is_empty():
                maker_order = level.peek_oldest()
                if not maker_order or not maker_order.is_active:
                    level.pop_oldest()
                    continue
                
                # Calculate trade quantity
                trade_qty = min(maker_order.remaining, incoming_order.remaining)
                
                # Execute trade at the resting order's price (price improvement for aggressor)
                trade = self._execute_trade(maker_order, incoming_order, trade_qty, level_price)
                trades.append(trade)
                
                # Remove maker order if fully filled
                if maker_order.remaining <= 0:
                    filled_order = level.pop_oldest()
                    self.orders.pop(maker_order.order_id, None)
                    if maker_order.client_order_id:
                        self.client_order_map.pop(maker_order.client_order_id, None)
            
            # Mark empty levels for removal
            if level.is_empty():
                levels_to_remove.append((key, level_price, incoming_order.side))
        
        # Clean up empty levels
        for key, price, side in levels_to_remove:
            opposite_side = "sell" if side == "buy" else "buy"
            self._remove_empty_level(price, opposite_side)
        
        return trades

    def submit_order(self, order: Order) -> Tuple[List[Trade], bool]:
        """
        Submit order to the book with comprehensive validation and matching.
        
        Returns:
            Tuple[List[Trade], bool]: (trades_executed, order_resting)
        """
        with self._lock:
            try:
                # Validate order
                if order.price:
                    self._validate_price(order.price)
                self._validate_quantity(order.remaining)
                
                # Update statistics
                self.stats["total_orders"] += 1
                
                trades = []
                order_resting = False
                
                # Attempt matching if marketable
                if self._is_order_marketable(order):
                    trades = self._match_order(order)
                
                # Handle remaining quantity based on order type
                if order.remaining > 0 and order.is_active:
                    if order.order_type == "market":
                        # Market order remainder cancelled (no more liquidity)
                        order.cancel()
                        logger.warning(f"Market order {order.order_id} partially cancelled: no liquidity")
                    
                    elif order.order_type == "ioc":
                        # IOC remainder cancelled
                        order.cancel()
                    
                    elif order.order_type == "fok":
                        # FOK should not reach here due to pre-check, but safety cancel
                        order.cancel()
                        trades = []  # Cancel all trades for FOK
                        logger.error(f"FOK order {order.order_id} inconsistent state")
                    
                    elif order.order_type == "limit":
                        # Rest limit order on book
                        self._add_resting_order(order)
                        order_resting = True
                
                return trades, order_resting
                
            except Exception as e:
                logger.error(f"Error submitting order {order.order_id}: {e}")
                order.reject(str(e))
                raise

    def get_order_status(self, order_id: str = None, client_order_id: str = None) -> Optional[Dict]:
        """Get current status of an order."""
        with self._lock:
            # Resolve client_order_id
            if client_order_id and not order_id:
                order_id = self.client_order_map.get(client_order_id)
            
            if not order_id or order_id not in self.orders:
                return None
            
            order, _ = self.orders[order_id]
            return order.to_dict()

    def get_statistics(self) -> Dict:
        """Get book statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                "active_orders": len(self.orders),
                "bid_levels": len(self.bids),
                "ask_levels": len(self.asks),
                "total_volume": str(stats["total_volume"]),
                "last_trade_price": str(stats["last_trade_price"]) if stats["last_trade_price"] else None
            })
            return stats

    # Legacy compatibility methods
    def top_n(self, n: int = 10) -> Dict:
        """Legacy method - use get_depth() instead."""
        return self.get_depth(n)

    def match(self, incoming: Order) -> List[Dict]:
        """Legacy method - use submit_order() instead."""
        trades, _ = self.submit_order(incoming)
        return [trade.to_dict() for trade in trades]

    def submit(self, order: Order) -> List[Dict]:
        """Legacy method - use submit_order() instead."""
        trades, _ = self.submit_order(order)
        return [trade.to_dict() for trade in trades]
    

    
