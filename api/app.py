# api/app.py
# CRITICAL: eventlet.monkey_patch() must be called FIRST, before any other imports
import eventlet
eventlet.monkey_patch()

import logging
import time
import json
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, disconnect
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, Optional
import threading
from functools import wraps
from collections import defaultdict, deque
import os
import sys

# Add the parent directory to the Python path to find the engine module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the engine modules
from engine.matcher import MatchingEngine, create_crypto_engine, MatchingEngineError
from engine.book import Trade
from engine.order import Order


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('matching_engine.log')
    ]
)
logger = logging.getLogger("matching-api")

# Create app & socketio
# Replace this section in your api/app.py (around line 40-50):

# Create app & socketio
app = Flask(__name__)
app.config.update({
    "SECRET_KEY": "your-secret-key-change-in-production",
    "DEBUG": False,
    "TESTING": False
})

# Configure static files IMMEDIATELY after creating the Flask app
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')
app.static_folder = static_dir
app.static_url_path = ''

print(f"Static folder configured: {app.static_folder}")  # Debug line to verify path

socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode="eventlet",
    logger=True,
    engineio_logger=False
)

# Create matching engine
engine = create_crypto_engine()

# Connection management
class ConnectionManager:
    """Manages WebSocket connections and subscriptions."""
    
    def __init__(self):
        self.connections: Dict[str, Dict] = {}  # session_id -> connection_info
        self.symbol_subscribers: Dict[str, set] = defaultdict(set)  # symbol -> set of session_ids
        self.trade_subscribers: set = set()  # session_ids subscribed to trades
        self.lock = threading.RLock()
    
    def add_connection(self, session_id: str, user_info: Dict = None):
        """Add a new connection."""
        with self.lock:
            self.connections[session_id] = {
                "connected_at": time.time(),
                "user_info": user_info or {},
                "subscribed_symbols": set(),
                "subscribed_to_trades": False,
                "message_count": 0
            }
    
    def remove_connection(self, session_id: str):
        """Remove connection and clean up subscriptions."""
        with self.lock:
            if session_id in self.connections:
                # Clean up symbol subscriptions
                for symbol in self.connections[session_id]["subscribed_symbols"]:
                    self.symbol_subscribers[symbol].discard(session_id)
                
                # Clean up trade subscription
                self.trade_subscribers.discard(session_id)
                
                del self.connections[session_id]
    
    def subscribe_to_symbol(self, session_id: str, symbol: str):
        """Subscribe connection to symbol updates."""
        with self.lock:
            if session_id in self.connections:
                self.symbol_subscribers[symbol].add(session_id)
                self.connections[session_id]["subscribed_symbols"].add(symbol)
    
    def unsubscribe_from_symbol(self, session_id: str, symbol: str):
        """Unsubscribe connection from symbol updates."""
        with self.lock:
            self.symbol_subscribers[symbol].discard(session_id)
            if session_id in self.connections:
                self.connections[session_id]["subscribed_symbols"].discard(symbol)
    
    def subscribe_to_trades(self, session_id: str):
        """Subscribe connection to trade feed."""
        with self.lock:
            self.trade_subscribers.add(session_id)
            if session_id in self.connections:
                self.connections[session_id]["subscribed_to_trades"] = True
    
    def get_symbol_subscribers(self, symbol: str) -> set:
        """Get all subscribers for a symbol."""
        with self.lock:
            return self.symbol_subscribers[symbol].copy()
    
    def get_trade_subscribers(self) -> set:
        """Get all trade subscribers."""
        with self.lock:
            return self.trade_subscribers.copy()
    
    def get_connection_count(self) -> int:
        """Get total connection count."""
        with self.lock:
            return len(self.connections)

# Global connection manager
conn_mgr = ConnectionManager()

# Rate limiting
class RateLimiter:
    """Simple token bucket rate limiter."""
    
    def __init__(self, max_requests: int = 10000, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.RLock()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier."""
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Clean old requests
            while self.requests[identifier] and self.requests[identifier][0] < window_start:
                self.requests[identifier].popleft()
            
            # Check limit
            if len(self.requests[identifier]) >= self.max_requests:
                return False
            
            # Add current request
            self.requests[identifier].append(now)
            return True

# Global rate limiter
rate_limiter = RateLimiter(max_requests=10000, window_seconds=60)

def rate_limit_required(f):
    """Rate limiting decorator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Use IP address as identifier
        identifier = request.environ.get('REMOTE_ADDR', 'unknown')
        
        if not rate_limiter.is_allowed(identifier):
            logger.warning(f"Rate limit exceeded for {identifier}")
            return jsonify({
                "error": "Rate limit exceeded",
                "error_code": "RATE_LIMIT",
                "retry_after": 60
            }), 429
        
        return f(*args, **kwargs)
    return decorated_function

def validate_json_request(required_fields: list = None):
    """Decorator to validate JSON requests."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    "error": "Content-Type must be application/json",
                    "error_code": "INVALID_CONTENT_TYPE"
                }), 400
            
            data = request.get_json()
            if not data:
                return jsonify({
                    "error": "Invalid JSON payload",
                    "error_code": "INVALID_JSON"
                }), 400
            
            if required_fields:
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return jsonify({
                        "error": f"Missing required fields: {missing_fields}",
                        "error_code": "MISSING_FIELDS"
                    }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Event handlers for engine
def handle_trade(trade: Trade):
    """Handle trade events from engine."""
    trade_data = trade.to_dict()
    
    # Emit to trade subscribers
    subscribers = conn_mgr.get_trade_subscribers()
    if subscribers:
        socketio.emit("trade", trade_data, namespace="/market")
        logger.debug(f"Emitted trade to {len(subscribers)} subscribers")

def handle_order_event(order: Order, event_type: str):
    """Handle order events from engine."""
    order_data = order.to_dict()
    order_data["event_type"] = event_type
    
    # Emit to symbol subscribers
    subscribers = conn_mgr.get_symbol_subscribers(order.symbol)
    if subscribers:
        socketio.emit("order_event", order_data, namespace="/market")
        logger.debug(f"Emitted order event to {len(subscribers)} subscribers")

# Register event handlers
engine.add_trade_handler(handle_trade)
engine.add_order_handler(handle_order_event)

# ---- Static File Route for Frontend ----
@app.route("/")
def index():
    """Serve the frontend."""
    return app.send_static_file('index.html')

@app.route("/<path:filename>")
def static_files(filename):
    """Serve static files."""
    return app.send_static_file(filename)

# ---- REST API Endpoints ----

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - engine.metrics["start_time"],
        "version": "2.0.0",
        "connections": conn_mgr.get_connection_count()
    }), 200

@app.route("/symbols", methods=["GET"])
def get_symbols():
    """Get list of supported trading symbols."""
    try:
        symbols = engine.get_symbols()
        symbol_info = {}
        
        for symbol in symbols:
            bbo = engine.get_bbo(symbol)
            stats = engine.get_statistics(symbol)
            symbol_info[symbol] = {
                "bbo": bbo,
                "active_orders": stats.get("active_orders", 0),
                "total_trades": stats.get("total_trades", 0)
            }
        
        return jsonify({
            "symbols": symbols,
            "symbol_info": symbol_info,
            "count": len(symbols)
        }), 200
        
    except Exception as e:
        logger.exception("Error fetching symbols")
        return jsonify({
            "error": "Failed to fetch symbols",
            "detail": str(e)
        }), 500

@app.route("/order", methods=["POST"])
@rate_limit_required
@validate_json_request(required_fields=["symbol", "side", "order_type", "quantity"])
def submit_order():
    """Submit a new order."""
    try:
        data = request.get_json()
        
        # Extract parameters
        symbol = data["symbol"]
        side = data["side"]
        order_type = data["order_type"]
        quantity = data["quantity"]
        price = data.get("price")
        client_order_id = data.get("client_order_id")
        
        logger.info(f"Order submission: {symbol} {side} {order_type} qty={quantity} price={price}")
        
        # Submit order to engine
        result = engine.submit_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            client_order_id=client_order_id
        )
        
        # Check for errors
        if "error" in result:
            status_code = 400 if result.get("error_code") == "VALIDATION_ERROR" else 500
            return jsonify(result), status_code
        
        # Emit market data updates
        try:
            # Emit L2 update to symbol subscribers
            snapshot = engine.get_book_snapshot(symbol, levels=10)
            subscribers = conn_mgr.get_symbol_subscribers(symbol)
            if subscribers:
                socketio.emit("l2_update", snapshot, namespace="/market")
        except Exception as e:
            logger.warning(f"Failed to emit market data update: {e}")
        
        return jsonify(result), 200
        
    except MatchingEngineError as e:
        logger.warning(f"Matching engine error: {e}")
        return jsonify({
            "error": str(e),
            "error_code": "ENGINE_ERROR"
        }), 400
        
    except Exception as e:
        logger.exception("Unexpected error in order submission")
        return jsonify({
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR"
        }), 500

@app.route("/cancel", methods=["POST"])
@rate_limit_required
@validate_json_request(required_fields=["symbol"])
def cancel_order():
    """Cancel an existing order."""
    try:
        data = request.get_json()
        
        symbol = data["symbol"]
        order_id = data.get("order_id")
        client_order_id = data.get("client_order_id")
        
        if not order_id and not client_order_id:
            return jsonify({
                "error": "Must provide order_id or client_order_id",
                "error_code": "MISSING_ORDER_ID"
            }), 400
        
        logger.info(f"Cancel request: symbol={symbol} order_id={order_id} client_order_id={client_order_id}")
        
        result = engine.cancel_order(symbol, order_id, client_order_id)
        
        # Emit market data update if successful
        if result.get("success"):
            try:
                snapshot = engine.get_book_snapshot(symbol, levels=10)
                subscribers = conn_mgr.get_symbol_subscribers(symbol)
                if subscribers:
                    socketio.emit("l2_update", snapshot, namespace="/market")
            except Exception as e:
                logger.warning(f"Failed to emit market data update after cancel: {e}")
        
        status_code = 200 if result.get("success") else 404
        return jsonify(result), status_code
        
    except Exception as e:
        logger.exception("Error in order cancellation")
        return jsonify({
            "error": "Failed to cancel order",
            "detail": str(e),
            "error_code": "CANCEL_ERROR"
        }), 500

@app.route("/order/<order_id>", methods=["GET"])
def get_order_status(order_id):
    """Get order status by order_id."""
    try:
        symbol = request.args.get("symbol")
        if not symbol:
            return jsonify({
                "error": "Symbol parameter is required",
                "error_code": "MISSING_SYMBOL"
            }), 400
        
        status = engine.get_order_status(symbol, order_id=order_id)
        
        if not status:
            return jsonify({
                "error": "Order not found",
                "error_code": "ORDER_NOT_FOUND"
            }), 404
        
        return jsonify(status), 200
        
    except Exception as e:
        logger.exception("Error fetching order status")
        return jsonify({
            "error": "Failed to fetch order status",
            "detail": str(e)
        }), 500

@app.route("/book/<symbol>", methods=["GET"])
def get_order_book(symbol):
    """Get order book snapshot."""
    try:
        levels = int(request.args.get("levels", 10))
        levels = min(max(levels, 1), 100)  # Limit between 1-100
        
        snapshot = engine.get_book_snapshot(symbol, levels)
        return jsonify(snapshot), 200
        
    except ValueError:
        return jsonify({
            "error": "Invalid levels parameter",
            "error_code": "INVALID_PARAMETER"
        }), 400
        
    except Exception as e:
        logger.exception("Error fetching order book")
        return jsonify({
            "error": "Failed to fetch order book",
            "detail": str(e)
        }), 500

@app.route("/bbo/<symbol>", methods=["GET"])
def get_best_bid_offer(symbol):
    """Get best bid/offer for symbol."""
    try:
        bbo = engine.get_bbo(symbol)
        return jsonify(bbo), 200
        
    except Exception as e:
        logger.exception("Error fetching BBO")
        return jsonify({
            "error": "Failed to fetch BBO",
            "detail": str(e)
        }), 500

@app.route("/statistics", methods=["GET"])
def get_statistics():
    """Get engine statistics."""
    try:
        symbol = request.args.get("symbol")
        stats = engine.get_statistics(symbol)
        return jsonify(stats), 200
        
    except Exception as e:
        logger.exception("Error fetching statistics")
        return jsonify({
            "error": "Failed to fetch statistics",
            "detail": str(e)
        }), 500

# ---- WebSocket Event Handlers ----

@socketio.on("connect", namespace="/market")
def on_connect():
    """Handle client connection."""
    session_id = request.sid
    user_agent = request.headers.get("User-Agent", "unknown")
    
    conn_mgr.add_connection(session_id, {"user_agent": user_agent})
    
    logger.info(f"Client connected: {session_id}")
    
    # Send welcome message with available symbols
    emit("connected", {
        "status": "connected",
        "session_id": session_id,
        "timestamp": time.time(),
        "symbols": engine.get_symbols()
    })

@socketio.on("disconnect", namespace="/market")
def on_disconnect():
    """Handle client disconnection."""
    session_id = request.sid
    conn_mgr.remove_connection(session_id)
    logger.info(f"Client disconnected: {session_id}")

@socketio.on("subscribe", namespace="/market")
def on_subscribe(data):
    """Handle subscription requests."""
    session_id = request.sid
    
    try:
        if not isinstance(data, dict):
            emit("error", {"message": "Invalid subscription data"})
            return
        
        subscription_type = data.get("type")
        symbol = data.get("symbol")
        
        if subscription_type == "l2_updates" and symbol:
            conn_mgr.subscribe_to_symbol(session_id, symbol.upper())
            
            # Send current snapshot
            snapshot = engine.get_book_snapshot(symbol, levels=10)
            emit("l2_update", snapshot)
            
            emit("subscribed", {
                "type": "l2_updates",
                "symbol": symbol.upper(),
                "timestamp": time.time()
            })
            
        elif subscription_type == "trades":
            conn_mgr.subscribe_to_trades(session_id)
            
            emit("subscribed", {
                "type": "trades",
                "timestamp": time.time()
            })
            
        else:
            emit("error", {"message": "Invalid subscription type or missing symbol"})
            
    except Exception as e:
        logger.exception("Error in subscription")
        emit("error", {"message": f"Subscription failed: {str(e)}"})

@socketio.on("unsubscribe", namespace="/market")
def on_unsubscribe(data):
    """Handle unsubscription requests."""
    session_id = request.sid
    
    try:
        if not isinstance(data, dict):
            emit("error", {"message": "Invalid unsubscription data"})
            return
        
        subscription_type = data.get("type")
        symbol = data.get("symbol")
        
        if subscription_type == "l2_updates" and symbol:
            conn_mgr.unsubscribe_from_symbol(session_id, symbol.upper())
            
            emit("unsubscribed", {
                "type": "l2_updates",
                "symbol": symbol.upper(),
                "timestamp": time.time()
            })
            
        elif subscription_type == "trades":
            conn_mgr.trade_subscribers.discard(session_id)
            
            emit("unsubscribed", {
                "type": "trades",
                "timestamp": time.time()
            })
            
        else:
            emit("error", {"message": "Invalid subscription type"})
            
    except Exception as e:
        logger.exception("Error in unsubscription")
        emit("error", {"message": f"Unsubscription failed: {str(e)}"})

@socketio.on("ping", namespace="/market")
def on_ping():
    """Handle ping requests."""
    emit("pong", {"timestamp": time.time()})

# ---- Error Handlers ----

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "error_code": "NOT_FOUND"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "error_code": "METHOD_NOT_ALLOWED"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    logger.exception("Internal server error")
    return jsonify({
        "error": "Internal server error",
        "error_code": "INTERNAL_ERROR"
    }), 500

# ---- Background Tasks ----

def periodic_cleanup():
    """Periodic cleanup of stale connections and data."""
    while True:
        try:
            # This would run cleanup tasks
            eventlet.sleep(300)  # Sleep for 5 minutes
            
            # Clean up stale connections, expired rate limits, etc.
            logger.debug("Running periodic cleanup")
            
        except Exception as e:
            logger.exception("Error in periodic cleanup")
            eventlet.sleep(60)

# Start background tasks
eventlet.spawn(periodic_cleanup)

# ---- Application Factory ----

def create_app(config=None):
    """Application factory pattern."""
    if config:
        app.config.update(config)
    
    return app

# ---- Run Server ----

def run(host="0.0.0.0", port=5000, debug=False):
    """Run the API server."""
    logger.info(f"Starting Matching Engine API server on {host}:{port}")
    logger.info(f"Supported symbols: {engine.get_symbols()}")
    logger.info(f"Debug mode: {debug}")
    
    socketio.run(
        app, 
        host=host, 
        port=port, 
        debug=debug,
        use_reloader=False  # Disable reloader to prevent duplicate processes
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cryptocurrency Matching Engine API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    run(host=args.host, port=args.port, debug=args.debug)