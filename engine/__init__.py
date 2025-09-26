# engine/__init__.py
"""
Enhanced matching engine core package.

This package contains the core matching engine components with enterprise-grade
features including:

- Thread-safe order book with price-time priority matching
- Support for multiple order types (Market, Limit, IOC, FOK)
- REG NMS-inspired trade-through protection
- High-performance data structures with O(log n) operations
- Comprehensive validation and error handling
- Fee calculation and trade reporting
- Event-driven architecture with callbacks

Components:
    - Order: Enhanced order representation with validation
    - OrderBook: Thread-safe order book with advanced matching
    - MatchingEngine: High-level engine coordinating multiple books
    - Trade: Trade execution records with fee calculations

Usage:
    from engine.matcher import MatchingEngine, create_crypto_engine
    from engine.order import Order, create_market_order, create_limit_order
    from engine.book import OrderBook
    
    # Create engine with crypto defaults
    engine = create_crypto_engine()
    
    # Submit orders
    result = engine.submit_order("BTC-USDT", "buy", "limit", "0.001", "30000")
"""

# Import only the classes that actually exist in the files
from .order import Order, create_market_order, create_limit_order
from .book import OrderBook, PriceLevel, Trade
from .matcher import (
    MatchingEngine, 
    MatchingEngineError,
    OrderValidationError,
    InsufficientLiquidityError,
    SymbolNotFoundError,
    FeeCalculator,
    SymbolConfig,
    create_crypto_engine,
    create_forex_engine
)

__version__ = "2.0.0"
__all__ = [
    # Order components
    "Order", "create_market_order", "create_limit_order",
    
    # Book components  
    "OrderBook", "PriceLevel", "Trade",
    
    # Engine components
    "MatchingEngine", "MatchingEngineError", "OrderValidationError", 
    "InsufficientLiquidityError", "SymbolNotFoundError",
    "FeeCalculator", "SymbolConfig",
    
    # Factory functions
    "create_crypto_engine", "create_forex_engine"
]

