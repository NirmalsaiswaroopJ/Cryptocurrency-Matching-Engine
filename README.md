# Cryptocurrency Matching Engine

A high-performance, REG NMS-inspired cryptocurrency matching engine built in Python with real-time WebSocket streaming and comprehensive order type support.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Features](#features)
- [Technical Requirements Met](#technical-requirements-met)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Performance Analysis](#performance-analysis)
- [Testing](#testing)
- [Contributing](#contributing)

## Problem Statement

The objective was to develop a high-performance cryptocurrency matching engine implementing core trading functionalities based on REG NMS-inspired principles. The system needed to:

### Core Requirements
1. **Price-Time Priority Matching**: Implement strict price-time priority with orders at better prices executing first, and at equal prices, earlier orders executing first (FIFO)
2. **Internal Order Protection**: Prevent trade-throughs by ensuring marketable orders always execute at the best available internal price
3. **Multiple Order Types**: Support Market, Limit, IOC (Immediate-or-Cancel), and FOK (Fill-or-Kill) orders
4. **Real-Time Market Data**: Provide live BBO (Best Bid/Offer) updates and L2 order book streaming
5. **High Performance**: Process >1000 orders per second with low latency
6. **Trade Execution Reporting**: Generate comprehensive trade data with audit trails

### Technical Specifications
- **Programming Language**: Python
- **Performance Target**: >1000 orders/sec
- **API Requirements**: REST endpoints + WebSocket streaming
- **Error Handling**: Robust validation and comprehensive logging
- **Code Quality**: Clean, maintainable, well-documented architecture

## Solution Overview

We developed a comprehensive trading system consisting of:

### Core Components
- **Matching Engine**: Thread-safe order matching with price-time priority
- **Order Book**: High-performance data structures using SortedDict for O(log n) operations
- **REST API**: Flask-based endpoints for order submission and market data queries
- **WebSocket Streaming**: Real-time market data dissemination via Socket.IO
- **Web Interface**: Professional trading interface with live order book and trade feed

### Key Innovations
- **Dual-Protocol Architecture**: REST for order management + WebSocket for real-time data
- **Event-Driven Design**: Asynchronous event emission for trade and order updates
- **Thread-Safe Implementation**: RLock-based concurrency control
- **Comprehensive Validation**: Multi-layer order parameter validation with caching
- **Professional UI**: Modern web interface with real-time market visualization

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │    REST API     │    │  WebSocket API  │
│   (React-like)  │    │   (Flask)       │    │  (Socket.IO)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     Matching Engine      │
                    │   (matcher.py)           │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      Order Books         │
                    │   (book.py)              │
                    │                          │
                    │  ┌─────────────────────┐ │
                    │  │   Price Levels      │ │
                    │  │   (SortedDict)      │ │
                    │  └─────────────────────┘ │
                    └──────────────────────────┘
```

## Features

### ✅ Completed Features

#### Core Matching Engine
- **Price-Time Priority**: Strict FIFO execution within price levels
- **Order Types**: Market, Limit, IOC, FOK with proper handling
- **Trade Generation**: Unique trade IDs with complete audit trails
- **BBO Calculation**: Real-time best bid/offer maintenance
- **Internal Order Protection**: Trade-through prevention

#### API Layer
- **REST Endpoints**: Order submission, cancellation, market data queries
- **WebSocket Streaming**: Real-time L2 updates, trade feed, order events
- **Rate Limiting**: Configurable request throttling (10,000 req/min default)
- **Error Handling**: Structured error responses with detailed codes
- **CORS Support**: Cross-origin resource sharing enabled

#### Web Interface
- **Live Order Book**: Real-time L2 market data display
- **Trade Feed**: Live trade execution stream
- **Order Entry**: Professional order placement interface
- **Market Statistics**: Real-time BBO, spread, and volume metrics
- **Connection Management**: WebSocket connection status and latency monitoring

#### Additional Features
- **Fee Calculation**: Maker-taker fee model implementation
- **Multiple Symbols**: Support for BTC-USDT, ETH-USDT, BNB-USDT, SOL-USDT
- **Performance Monitoring**: Comprehensive metrics and statistics
- **Load Testing**: Advanced load testing suite with monitoring
- **Logging**: Detailed audit trails for all operations

## Technical Requirements Met

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| **Python Implementation** | ✅ Complete | Pure Python with Flask/Socket.IO |
| **>1000 orders/sec** | ⚠️ Partial | Currently ~375 orders/sec (optimization needed) |
| **Robust Error Handling** | ✅ Complete | Multi-layer validation, structured errors |
| **Comprehensive Logging** | ✅ Complete | Debug, info, warning, error levels |
| **Clean Architecture** | ✅ Complete | Modular design, separation of concerns |
| **Unit Tests** | ❌ Missing | Testing framework needed |
| **Advanced Order Types** | ✅ Complete | Market, Limit, IOC, FOK |
| **Persistence** | ❌ Not Implemented | In-memory only currently |
| **Performance Optimization** | ⚠️ In Progress | Some optimizations, more needed |
| **Fee Model** | ✅ Complete | Maker-taker fees with reporting |

## Project Structure

```
GOQUANT/
├── api/                          # REST API and WebSocket server
│   ├── __init__.py
│   └── app.py                    # Main Flask application
├── engine/                       # Core matching engine
│   ├── __init__.py
│   ├── book.py                   # Order book implementation
│   ├── matcher.py                # Matching engine logic
│   └── order.py                  # Order definitions
├── frontend/                     # Web interface
│   ├── index.html                # Trading interface
│   └── main.js                   # Frontend JavaScript
├── tests/                        # Testing suite
│   ├── advanced_load_test.py     # Advanced load testing
│   └── monitoring_suite.py       # Performance monitoring
├── Documentation/                # Technical documentation
└── README.md                     # This file
```

### Key Files Description

#### Core Engine (`engine/`)
- **`matcher.py`**: Central matching engine with order validation, trade execution, and event management
- **`book.py`**: Order book implementation with SortedDict-based price levels and matching logic
- **`order.py`**: Order class definition with validation and state management

#### API Layer (`api/`)
- **`app.py`**: Flask application with REST endpoints, WebSocket handlers, and connection management

#### Frontend (`frontend/`)
- **`index.html`**: Professional trading interface with real-time market data
- **`main.js`**: WebSocket client with order submission and market data handling

#### Testing (`tests/`)
- **`advanced_load_test.py`**: Comprehensive load testing with performance metrics
- **`monitoring_suite.py`**: System monitoring and profiling tools

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Dependencies
```bash
pip install flask flask-socketio eventlet sortedcontainers redis
```

### Quick Start
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GOQUANT
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt  # or install manually as shown above
   ```

3. **Start the server**
   ```bash
   python api/app.py
   ```

4. **Access the interface**
   - Open http://localhost:5000 in your browser
   - The trading interface should load with connection established

### Alternative Startup Methods
```bash
# With custom host/port
python api/app.py --host 0.0.0.0 --port 8080

# With debug mode
python api/app.py --debug
```

## Usage

### Web Interface
1. **Connect**: Interface automatically connects via WebSocket
2. **Place Orders**: Use the order entry form on the left panel
3. **Monitor Market**: Watch real-time order book and trade feed
4. **Cancel Orders**: Use order ID to cancel existing orders

### REST API Examples

#### Submit Order
```bash
curl -X POST http://localhost:5000/order \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC-USDT",
    "side": "buy",
    "order_type": "limit",
    "quantity": "0.001",
    "price": "30000.00"
  }'
```

#### Get Market Data
```bash
# Order book snapshot
curl http://localhost:5000/book/BTC-USDT?levels=5

# Best bid/offer
curl http://localhost:5000/bbo/BTC-USDT

# Statistics
curl http://localhost:5000/statistics
```

### WebSocket Usage
```javascript
const socket = io("/market");

// Subscribe to market data
socket.emit("subscribe", {
  type: "l2_updates",
  symbol: "BTC-USDT"
});

// Subscribe to trades
socket.emit("subscribe", {
  type: "trades"
});

// Handle updates
socket.on("l2_update", (data) => {
  console.log("Order book update:", data);
});

socket.on("trade", (trade) => {
  console.log("New trade:", trade);
});
```

## API Documentation

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health and uptime |
| GET | `/symbols` | Available trading pairs |
| POST | `/order` | Submit new order |
| POST | `/cancel` | Cancel existing order |
| GET | `/book/{symbol}` | Order book snapshot |
| GET | `/bbo/{symbol}` | Best bid/offer |
| GET | `/statistics` | Engine statistics |

### WebSocket Events

| Event | Description |
|-------|-------------|
| `connect` | Client connection established |
| `subscribe` | Subscribe to data feeds |
| `l2_update` | Order book updates |
| `trade` | Trade execution events |
| `order_event` | Order status changes |

For detailed API specification, see [Documentation/CME_API Specification.pdf](Documentation).

## Performance Analysis

### Current Performance Metrics
- **Throughput**: ~375 orders/second (target: 1000+)
- **Latency**: 
  - Average: ~2-5ms for successful orders
  - P99: ~15ms under normal load
  - P99.9: High latency spikes under stress (optimization needed)
- **Success Rate**: 94-95% under load testing
- **Memory Usage**: Stable, no significant leaks detected

### Performance Bottlenecks Identified
1. **Global Locking**: Single engine lock creates contention
2. **Synchronous Event Emission**: Blocks order processing
3. **Order Cancellation**: O(n) operation within price levels
4. **Thread Pool Limitations**: Flask threading model constraints

### Optimization Roadmap
1. **Per-Symbol Locking**: Eliminate global contention
2. **Asynchronous Events**: Decouple event emission from processing
3. **Lock-Free Structures**: Implement for read-heavy operations
4. **Connection Pooling**: Optimize WebSocket management

## Testing

### Load Testing
```bash
# Basic load test
python tests/advanced_load_test.py --rate 500 --duration 60

# High-frequency test
python tests/advanced_load_test.py --rate 1000 --duration 120

# System monitoring during test
python tests/monitoring_suite.py --test comprehensive
```

### Performance Monitoring
```bash
# Real-time system monitoring
python tests/monitoring_suite.py --test monitor

# Generate performance report
python tests/monitoring_suite.py --test profile --iterations 1000
```

### Expected Results
- **Success Rate**: >95%
- **Average Latency**: <10ms
- **Throughput**: Target 1000+ orders/sec (current ~375)

## Troubleshooting

### Common Issues

1. **"Endpoint not found" error**
   - Check static file configuration in app.py
   - Ensure frontend files are in correct directory structure

2. **WebSocket connection fails**
   - Verify Socket.IO client version compatibility
   - Check CORS settings in app.py

3. **Performance degradation**
   - Monitor system resources during load
   - Check for thread contention in logs
   - Consider reducing concurrent connections

4. **Order validation errors**
   - Verify decimal precision in order parameters
   - Check symbol configuration limits
   - Ensure price/quantity formats are correct

### Debug Mode
```bash
python api/app.py --debug
```

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Follow existing code style and patterns
4. Add tests for new functionality
5. Submit pull request with detailed description

### Code Standards
- Use type hints where possible
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include error handling and logging
- Write unit tests for core functionality

### Priority Areas for Contribution
1. **Unit Testing**: Comprehensive test suite needed
2. **Performance Optimization**: Address throughput bottlenecks
3. **Persistence Layer**: Add database integration
4. **Advanced Order Types**: Stop-loss, iceberg orders
5. **Monitoring Dashboard**: Enhanced system monitoring

## License

This project is developed for educational and research purposes. See LICENSE file for details.

## Acknowledgments

Built following REG NMS principles and modern electronic trading system design patterns. Special consideration given to market microstructure theory and high-frequency trading requirements.

---

**Note**: This system demonstrates core matching engine concepts and is suitable for educational and development use. For production deployment, additional security, monitoring, and regulatory compliance features would be required.
