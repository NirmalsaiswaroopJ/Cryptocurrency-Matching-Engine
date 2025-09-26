# api/__init__.py
"""
Enhanced API package for the cryptocurrency matching engine.

This package provides a comprehensive Flask-based REST API with WebSocket support
for real-time market data streaming. Features include:

- Order submission and cancellation
- Real-time market data feeds
- Connection management and rate limiting  
- Comprehensive error handling and logging
- Performance monitoring and statistics

Usage:
    from api.app import create_app, run
    
    app = create_app()
    run(host="0.0.0.0", port=5000)
"""

from .app import create_app, run, engine, socketio

__version__ = "2.0.0"
__all__ = ["create_app", "run", "engine", "socketio"]