# engine/matcher.py
from decimal import Decimal, InvalidOperation
from engine.book import OrderBook, Trade
from engine.order import Order
from typing import Dict, Any, Optional, List, Callable
import threading
import logging
import time
from collections import defaultdict


logger = logging.getLogger(__name__)


class MatchingEngineError(Exception):
    """Base exception for matching engine errors."""
    pass


class OrderValidationError(MatchingEngineError):
    """Raised when order validation fails."""
    pass


class InsufficientLiquidityError(MatchingEngineError):
    """Raised when there's insufficient liquidity for FOK orders."""
    pass


class SymbolNotFoundError(MatchingEngineError):
    """Raised when symbol is not supported."""
    pass


class FeeCalculator:
    """Simple maker-taker fee calculator."""
    
    def __init__(self, maker_fee: Decimal = Decimal("0.001"), taker_fee: Decimal = Decimal("0.001")):
        self.maker_fee = maker_fee  # 0.1% default
        self.taker_fee = taker_fee  # 0.1% default
    
    def calculate_maker_fee(self, trade_value: Decimal) -> Decimal:
        """Calculate maker fee for a trade."""
        return trade_value * self.maker_fee
    
    def calculate_taker_fee(self, trade_value: Decimal) -> Decimal:
        """Calculate taker fee for a trade."""
        return trade_value * self.taker_fee
    
    def calculate_fees(self, trade: Trade) -> Dict[str, Decimal]:
        """Calculate both maker and taker fees for a trade."""
        trade_value = trade.price * trade.quantity
        return {
            "maker_fee": self.calculate_maker_fee(trade_value),
            "taker_fee": self.calculate_taker_fee(trade_value),
            "trade_value": trade_value
        }


class SymbolConfig:
    """Configuration for a trading symbol."""
    
    def __init__(self, symbol: str, tick_size: Decimal = Decimal("0.01"), 
                 min_quantity: Decimal = Decimal("0.00000001"),
                 max_quantity: Decimal = Decimal("1000000"),
                 min_notional: Decimal = Decimal("10")):
        self.symbol = symbol
        self.tick_size = tick_size
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
        self.min_notional = min_notional  # Minimum order value


class MatchingEngine:
    """
    High-performance matching engine optimized for low latency and high throughput.
    Simplified design focused on core functionality.
    """
    
    def __init__(self, fee_calculator: FeeCalculator = None):
        # Order books by symbol
        self.books: Dict[str, OrderBook] = {}
        
        # Symbol configurations
        self.symbol_configs: Dict[str, SymbolConfig] = {}
        
        # Fee calculator
        self.fee_calculator = fee_calculator or FeeCalculator()
        
        # Thread safety - single lock for simplicity and correctness
        self._lock = threading.RLock()
        
        # Event handlers - simple synchronous callbacks
        self.trade_handlers: List[Callable[[Trade], None]] = []
        self.order_handlers: List[Callable[[Order, str], None]] = []
        
        # Performance metrics
        self.metrics = {
            "orders_processed": 0,
            "trades_executed": 0,
            "total_volume": Decimal("0"),
            "avg_latency_ms": 0.0,
            "start_time": time.time()
        }
        
        # Error tracking
        self.error_counts = defaultdict(int)
        
        logger.info("MatchingEngine initialized")

    def add_symbol(self, symbol: str, config: SymbolConfig = None) -> None:
        """Add a new trading symbol with optional configuration."""
        with self._lock:
            if symbol in self.books:
                logger.warning(f"Symbol {symbol} already exists")
                return
            
            if not config:
                config = SymbolConfig(symbol)
            
            self.symbol_configs[symbol] = config
            self.books[symbol] = OrderBook(
                symbol=symbol,
                tick_size=config.tick_size,
                min_quantity=config.min_quantity
            )
            
            logger.info(f"Added symbol {symbol} with tick_size={config.tick_size}")

    def remove_symbol(self, symbol: str) -> bool:
        """Remove a trading symbol. Returns True if removed."""
        with self._lock:
            if symbol not in self.books:
                return False
            
            # Check for active orders
            book = self.books[symbol]
            if len(book.orders) > 0:
                logger.warning(f"Cannot remove symbol {symbol}: has active orders")
                return False
            
            del self.books[symbol]
            del self.symbol_configs[symbol]
            logger.info(f"Removed symbol {symbol}")
            return True

    def get_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        with self._lock:
            return list(self.books.keys())

    def _get_book(self, symbol: str) -> OrderBook:
        """Get order book for symbol, creating if necessary."""
        if symbol not in self.books:
            # Auto-create with default config
            self.add_symbol(symbol)
        return self.books[symbol]

    def _validate_order_params(self, symbol: str, side: str, order_type: str, 
                              quantity, price: Optional[float] = None) -> tuple:
        """Validate and normalize order parameters."""
        # Normalize strings
        symbol = symbol.upper().strip()
        side = side.lower().strip()
        order_type = order_type.lower().strip()
        
        # Validate basic params
        if not symbol:
            raise OrderValidationError("Symbol cannot be empty")
        
        if side not in ("buy", "sell"):
            raise OrderValidationError("Side must be 'buy' or 'sell'")
        
        if order_type not in ("market", "limit", "ioc", "fok"):
            raise OrderValidationError("Invalid order type")
        
        # Convert and validate quantity
        try:
            quantity = Decimal(str(quantity))
            if quantity <= 0:
                raise OrderValidationError("Quantity must be positive")
        except (InvalidOperation, TypeError, ValueError) as e:
            raise OrderValidationError(f"Invalid quantity: {e}")
        
        # Convert and validate price
        validated_price = None
        if price is not None:
            try:
                validated_price = Decimal(str(price))
                if validated_price <= 0:
                    raise OrderValidationError("Price must be positive")
            except (InvalidOperation, TypeError, ValueError) as e:
                raise OrderValidationError(f"Invalid price: {e}")
        
        # Check symbol-specific limits
        if symbol in self.symbol_configs:
            config = self.symbol_configs[symbol]
            
            if quantity < config.min_quantity:
                raise OrderValidationError(f"Quantity below minimum: {config.min_quantity}")
            
            if quantity > config.max_quantity:
                raise OrderValidationError(f"Quantity above maximum: {config.max_quantity}")
            
            if validated_price and config.min_notional:
                notional = validated_price * quantity
                if notional < config.min_notional:
                    raise OrderValidationError(f"Order value below minimum: {config.min_notional}")
        
        return symbol, side, order_type, quantity, validated_price

    def add_trade_handler(self, handler: Callable[[Trade], None]) -> None:
        """Add a trade event handler."""
        self.trade_handlers.append(handler)

    def add_order_handler(self, handler: Callable[[Order, str], None]) -> None:
        """Add an order event handler."""
        self.order_handlers.append(handler)

    def _emit_trade_event(self, trade: Trade) -> None:
        """Emit trade event to all handlers."""
        for handler in self.trade_handlers:
            try:
                handler(trade)
            except Exception as e:
                logger.error(f"Trade handler error: {e}")

    def _emit_order_event(self, order: Order, event_type: str) -> None:
        """Emit order event to all handlers."""
        for handler in self.order_handlers:
            try:
                handler(order, event_type)
            except Exception as e:
                logger.error(f"Order handler error: {e}")

    def submit_order(self, symbol: str, side: str, order_type: str, 
                    quantity, price: Optional[float] = None, 
                    client_order_id: str = None) -> Dict[str, Any]:
        """
        Submit an order with comprehensive validation and error handling.
        
        Returns:
            Dict containing order_id, trades, fees, and market data
        """
        start_time = time.time()
        
        try:
            # Validate parameters
            symbol, side, order_type, quantity, validated_price = self._validate_order_params(
                symbol, side, order_type, quantity, price
            )
            
            # Create order
            order = Order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=validated_price,
                client_order_id=client_order_id
            )
            
            # Process order with single lock
            with self._lock:
                # Get book and submit order
                book = self._get_book(symbol)
                trades, order_resting = book.submit_order(order)
                
                # Calculate fees for trades
                trade_dicts = []
                total_fees = {"maker_fees": Decimal("0"), "taker_fees": Decimal("0")}
                
                for trade in trades:
                    trade_dict = trade.to_dict()
                    
                    # Add fee information
                    fees = self.fee_calculator.calculate_fees(trade)
                    trade_dict.update({
                        "maker_fee": str(fees["maker_fee"]),
                        "taker_fee": str(fees["taker_fee"]),
                        "trade_value": str(fees["trade_value"])
                    })
                    
                    total_fees["maker_fees"] += fees["maker_fee"]
                    total_fees["taker_fees"] += fees["taker_fee"]
                    
                    trade_dicts.append(trade_dict)
                    
                    # Emit trade event
                    self._emit_trade_event(trade)
                
                # Emit order events
                if trades:
                    if order.remaining <= 0:
                        self._emit_order_event(order, "filled")
                    else:
                        self._emit_order_event(order, "partially_filled")
                
                if order_resting:
                    self._emit_order_event(order, "resting")
                
                if order.status == "cancelled":
                    self._emit_order_event(order, "cancelled")
                
                if order.status == "rejected":
                    self._emit_order_event(order, "rejected")
                
                # Update metrics
                self.metrics["orders_processed"] += 1
                self.metrics["trades_executed"] += len(trades)
                if trades:
                    for trade in trades:
                        self.metrics["total_volume"] += trade.quantity
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                self.metrics["avg_latency_ms"] = (
                    (self.metrics["avg_latency_ms"] * (self.metrics["orders_processed"] - 1) + latency_ms) /
                    self.metrics["orders_processed"]
                )
                
                # Build response
                response = {
                    "order_id": order.order_id,
                    "client_order_id": order.client_order_id,
                    "status": order.status,
                    "filled_quantity": str(order.filled),
                    "remaining_quantity": str(order.remaining),
                    "trades": trade_dicts,
                    "total_maker_fees": str(total_fees["maker_fees"]),
                    "total_taker_fees": str(total_fees["taker_fees"]),
                    "bbo": book.get_bbo(),
                    "processing_time_ms": latency_ms
                }
                
                logger.debug(f"Order submitted: {order.order_id} {side} {quantity}@{validated_price or 'MKT'} -> {len(trades)} trades")
                return response
            
        except Exception as e:
            self.error_counts[type(e).__name__] += 1
            logger.error(f"Order submission failed: {e}")
            
            error_response = {
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            
            if isinstance(e, (OrderValidationError, InsufficientLiquidityError)):
                error_response["error_code"] = "VALIDATION_ERROR"
            else:
                error_response["error_code"] = "INTERNAL_ERROR"
            
            return error_response

    def cancel_order(self, symbol: str, order_id: str = None, 
                    client_order_id: str = None) -> Dict[str, Any]:
        """Cancel an order by order_id or client_order_id."""
        start_time = time.time()
        
        try:
            if not order_id and not client_order_id:
                raise OrderValidationError("Must provide order_id or client_order_id")
            
            symbol = symbol.upper().strip()
            
            with self._lock:
                book = self._get_book(symbol)
                
                # Get order status before cancellation
                order_status = book.get_order_status(order_id, client_order_id)
                if not order_status:
                    return {
                        "success": False,
                        "error": "Order not found",
                        "processing_time_ms": (time.time() - start_time) * 1000
                    }
                
                # Cancel the order
                success = book.cancel_order(order_id, client_order_id)
                
                if success:
                    # Create order object for event
                    cancelled_order = Order(
                        symbol=symbol,
                        side=order_status["side"],
                        order_type=order_status["order_type"],
                        quantity=Decimal(order_status["quantity"]),
                        price=Decimal(order_status["price"]) if order_status["price"] else None
                    )
                    cancelled_order.order_id = order_status["order_id"]
                    cancelled_order.status = "cancelled"
                    
                    self._emit_order_event(cancelled_order, "cancelled")
                    logger.debug(f"Order cancelled: {order_status['order_id']}")
                
                return {
                    "success": success,
                    "order_id": order_status["order_id"],
                    "client_order_id": order_status.get("client_order_id"),
                    "bbo": book.get_bbo(),
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            
        except Exception as e:
            self.error_counts[type(e).__name__] += 1
            logger.error(f"Order cancellation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000
            }

    def get_order_status(self, symbol: str, order_id: str = None, 
                        client_order_id: str = None) -> Optional[Dict]:
        """Get order status."""
        try:
            symbol = symbol.upper().strip()
            with self._lock:
                if symbol not in self.books:
                    return None
                
                book = self.books[symbol]
                return book.get_order_status(order_id, client_order_id)
                
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None

    def get_book_snapshot(self, symbol: str, levels: int = 10) -> Dict:
        """Get L2 order book snapshot."""
        try:
            symbol = symbol.upper().strip()
            with self._lock:
                book = self._get_book(symbol)
                return book.get_depth(levels)
                
        except Exception as e:
            logger.error(f"Error getting book snapshot: {e}")
            return {
                "timestamp": time.time(),
                "symbol": symbol,
                "error": str(e),
                "bids": [],
                "asks": []
            }

    def get_bbo(self, symbol: str) -> Dict:
        """Get Best Bid Offer for symbol."""
        try:
            symbol = symbol.upper().strip()
            with self._lock:
                book = self._get_book(symbol)
                return book.get_bbo()
                
        except Exception as e:
            logger.error(f"Error getting BBO: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "best_bid": None,
                "best_ask": None,
                "timestamp": time.time()
            }

    def get_statistics(self, symbol: str = None) -> Dict:
        """Get engine or symbol-specific statistics."""
        try:
            with self._lock:
                if symbol:
                    symbol = symbol.upper().strip()
                    if symbol not in self.books:
                        return {"error": f"Symbol {symbol} not found"}
                    return self.books[symbol].get_statistics()
                else:
                    # Engine-wide statistics
                    uptime = time.time() - self.metrics["start_time"]
                    stats = self.metrics.copy()
                    stats.update({
                        "uptime_seconds": uptime,
                        "symbols_count": len(self.books),
                        "total_volume": str(stats["total_volume"]),
                        "orders_per_second": stats["orders_processed"] / max(uptime, 1),
                        "error_counts": dict(self.error_counts)
                    })
                    return stats
                    
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    def reset_statistics(self) -> None:
        """Reset performance metrics."""
        with self._lock:
            self.metrics = {
                "orders_processed": 0,
                "trades_executed": 0,
                "total_volume": Decimal("0"),
                "avg_latency_ms": 0.0,
                "start_time": time.time()
            }
            self.error_counts.clear()
            logger.info("Statistics reset")

    # Legacy compatibility methods
    def cancel(self, symbol: str, order_id: str) -> bool:
        """Legacy method - use cancel_order() instead."""
        result = self.cancel_order(symbol, order_id=order_id)
        return result.get("success", False)

    def top_n(self, symbol: str, n: int = 10) -> Dict:
        """Legacy method - use get_book_snapshot() instead."""
        return self.get_book_snapshot(symbol, n)


# Convenience factory functions
def create_crypto_engine() -> MatchingEngine:
    """Create a matching engine with common crypto symbol configurations."""
    engine = MatchingEngine()
    
    # Add common crypto pairs
    crypto_symbols = {
        "BTC-USDT": SymbolConfig("BTC-USDT", Decimal("0.01"), Decimal("0.00001")),
        "ETH-USDT": SymbolConfig("ETH-USDT", Decimal("0.01"), Decimal("0.0001")),
        "BNB-USDT": SymbolConfig("BNB-USDT", Decimal("0.01"), Decimal("0.001")),
        "SOL-USDT": SymbolConfig("SOL-USDT", Decimal("0.01"), Decimal("0.01")),
    }
    
    for symbol, config in crypto_symbols.items():
        engine.add_symbol(symbol, config)
    
    return engine


def create_forex_engine() -> MatchingEngine:
    """Create a matching engine with common forex pair configurations."""
    engine = MatchingEngine(FeeCalculator(Decimal("0.0001"), Decimal("0.0001")))  # Lower fees for forex
    
    # Add major forex pairs
    forex_symbols = {
        "EUR-USD": SymbolConfig("EUR-USD", Decimal("0.00001"), Decimal("1000")),
        "GBP-USD": SymbolConfig("GBP-USD", Decimal("0.00001"), Decimal("1000")),
        "USD-JPY": SymbolConfig("USD-JPY", Decimal("0.001"), Decimal("1000")),
    }
    
    for symbol, config in forex_symbols.items():
        engine.add_symbol(symbol, config)
    
    return engine
                