// frontend/main.js
// Enhanced cryptocurrency matching engine frontend with advanced features

(() => {
    // Configuration
    const CONFIG = {
        API_HOST: "", // empty => same origin
        SOCKET_PATH: "/socket.io/",
        NAMESPACE: "/market",
        RECONNECT_ATTEMPTS: 5,
        RECONNECT_DELAY: 2000,
        PING_INTERVAL: 30000,
        MAX_TRADE_HISTORY: 100,
        MAX_ORDERBOOK_LEVELS: 20
    };

    // State management
    const state = {
        connected: false,
        currentSymbol: "BTC-USDT",
        orderCount: 0,
        tradeCount: 0,
        latency: 0,
        lastPingTime: 0,
        subscriptions: new Set(),
        orderBook: {
            bids: [],
            asks: [],
            lastUpdate: 0
        },
        trades: [],
        statistics: {
            spread: null,
            midPrice: null,
            volume24h: 0
        }
    };

    // DOM elements
    const elements = {
        // Connection status
        connStatus: document.getElementById("conn-status"),
        latency: document.getElementById("latency"),
        orderCount: document.getElementById("order-count"),
        tradeCount: document.getElementById("trade-count"),

        // Order form
        symbol: document.getElementById("symbol"),
        side: document.getElementById("side"),
        orderType: document.getElementById("order_type"),
        quantity: document.getElementById("quantity"),
        price: document.getElementById("price"),
        clientOrderId: document.getElementById("client_order_id"),
        submitBtn: document.getElementById("submitBtn"),
        orderAlert: document.getElementById("order-alert"),

        // Cancel form
        cancelOrderId: document.getElementById("cancel_order_id"),
        cancelBtn: document.getElementById("cancelBtn"),

        // Market data
        currentSymbol: document.getElementById("current-symbol"),
        bestBid: document.getElementById("best_bid"),
        bestAsk: document.getElementById("best_ask"),
        spread: document.getElementById("spread"),
        midPrice: document.getElementById("mid_price"),
        asksTableBody: document.getElementById("asks_table_body"),
        bidsTableBody: document.getElementById("bids_table_body"),

        // Trade feed
        tradeFeed: document.getElementById("trade_feed"),
        clearTrades: document.getElementById("clear-trades"),
        toastContainer: document.getElementById("toast-container")
    };

    // Socket.IO client
    let socket = null;
    let reconnectAttempts = 0;
    let pingInterval = null;

    // Initialize application
    function init() {
        setupEventListeners();
        connectSocket();
        startPingTimer();
        loadSymbols();
        
        // Set initial symbol
        updateSymbolDisplay();
        
        console.info("Matching Engine Frontend initialized");
    }

    // Socket connection management
    function connectSocket() {
        try {
            socket = io(CONFIG.API_HOST + CONFIG.NAMESPACE, {
                path: CONFIG.SOCKET_PATH,
                transports: ['websocket', 'polling'],
                timeout: 5000,
                forceNew: true
            });

            socket.on("connect", onSocketConnect);
            socket.on("disconnect", onSocketDisconnect);
            socket.on("connect_error", onSocketError);
            socket.on("reconnect", onSocketReconnect);
            
            // Market data events
            socket.on("connected", onWelcome);
            socket.on("l2_update", onL2Update);
            socket.on("trade", onTrade);
            socket.on("order_event", onOrderEvent);
            socket.on("subscribed", onSubscribed);
            socket.on("error", onSocketServerError);
            socket.on("pong", onPong);

        } catch (error) {
            console.error("Failed to initialize socket:", error);
            showToast("Connection failed", "error");
        }
    }

    // Socket event handlers
    function onSocketConnect() {
        console.info("Connected to matching engine:", socket.id);
        state.connected = true;
        reconnectAttempts = 0;
        updateConnectionStatus();
        
        // Subscribe to current symbol
        subscribeToSymbol(state.currentSymbol);
        subscribeToTrades();
        
        showToast("Connected to matching engine", "success");
    }

    function onSocketDisconnect(reason) {
        console.warn("Disconnected from server:", reason);
        state.connected = false;
        updateConnectionStatus();
        showToast("Connection lost", "error");

        // Attempt reconnection for client-side disconnects
        if (reason !== "io server disconnect" && reconnectAttempts < CONFIG.RECONNECT_ATTEMPTS) {
            setTimeout(() => {
                reconnectAttempts++;
                console.info(`Reconnection attempt ${reconnectAttempts}/${CONFIG.RECONNECT_ATTEMPTS}`);
                connectSocket();
            }, CONFIG.RECONNECT_DELAY * reconnectAttempts);
        }
    }

    function onSocketError(error) {
        console.error("Socket connection error:", error);
        showToast(`Connection error: ${error.message}`, "error");
    }

    function onSocketReconnect() {
        console.info("Reconnected to server");
        showToast("Reconnected successfully", "success");
    }

    function onWelcome(data) {
        console.info("Welcome message:", data);
        if (data.symbols && Array.isArray(data.symbols)) {
            updateSymbolOptions(data.symbols);
        }
    }

    function onL2Update(data) {
        if (!data || data.symbol !== state.currentSymbol) return;

        state.orderBook = {
            bids: data.bids || [],
            asks: data.asks || [],
            lastUpdate: data.timestamp || Date.now() / 1000
        };

        updateOrderBookDisplay();
        updateBBODisplay(data);
        updateStatistics();
    }

    function onTrade(trade) {
        if (!trade) return;

        state.tradeCount++;
        elements.tradeCount.textContent = state.tradeCount;

        // Add to trade history
        state.trades.unshift(trade);
        if (state.trades.length > CONFIG.MAX_TRADE_HISTORY) {
            state.trades = state.trades.slice(0, CONFIG.MAX_TRADE_HISTORY);
        }

        // Update display
        addTradeToFeed(trade);
        
        // Play sound notification (optional)
        playTradeSound();
    }

    function onOrderEvent(orderData) {
        console.info("Order event:", orderData);
        // Could be used to update order status, show notifications, etc.
    }

    function onSubscribed(data) {
        console.info("Subscription confirmed:", data);
        state.subscriptions.add(data.type + (data.symbol ? ":" + data.symbol : ""));
    }

    function onSocketServerError(error) {
        console.error("Server error:", error);
        showToast(`Server error: ${error.message}`, "error");
    }

    function onPong(data) {
        if (state.lastPingTime > 0) {
            state.latency = Date.now() - state.lastPingTime;
            elements.latency.textContent = `${state.latency}ms`;
        }
    }

    // Subscription management
    function subscribeToSymbol(symbol) {
        if (socket && socket.connected) {
            socket.emit("subscribe", {
                type: "l2_updates",
                symbol: symbol
            });
        }
    }

    function subscribeToTrades() {
        if (socket && socket.connected) {
            socket.emit("subscribe", {
                type: "trades"
            });
        }
    }

    function unsubscribeFromSymbol(symbol) {
        if (socket && socket.connected) {
            socket.emit("unsubscribe", {
                type: "l2_updates",
                symbol: symbol
            });
        }
    }

    // UI Update functions
    function updateConnectionStatus() {
        if (state.connected) {
            elements.connStatus.textContent = "Connected";
            elements.connStatus.className = "status-value status-connected";
        } else {
            elements.connStatus.textContent = "Disconnected";
            elements.connStatus.className = "status-value status-disconnected";
            elements.latency.textContent = "-- ms";
        }
    }

    function updateSymbolDisplay() {
        elements.currentSymbol.textContent = state.currentSymbol;
    }

    function updateOrderBookDisplay() {
        const { bids, asks } = state.orderBook;

        // Clear existing data
        elements.asksTableBody.innerHTML = "";
        elements.bidsTableBody.innerHTML = "";

        // Add asks (reverse order to show best ask at bottom)
        const reversedAsks = [...asks].reverse().slice(0, CONFIG.MAX_ORDERBOOK_LEVELS);
        reversedAsks.forEach(([price, size]) => {
            const total = calculateRunningTotal(reversedAsks, price, true);
            addOrderBookRow(elements.asksTableBody, price, size, total, "asks");
        });

        // Add bids (best bid at top)
        const topBids = bids.slice(0, CONFIG.MAX_ORDERBOOK_LEVELS);
        topBids.forEach(([price, size]) => {
            const total = calculateRunningTotal(topBids, price, false);
            addOrderBookRow(elements.bidsTableBody, price, size, total, "bids");
        });
    }

    function addOrderBookRow(tableBody, price, size, total, side) {
        const row = document.createElement("tr");
        row.className = `orderbook-row ${side}`;

        row.innerHTML = `
            <td class="orderbook-cell">${formatPrice(price)}</td>
            <td class="orderbook-cell">${formatQuantity(size)}</td>
            <td class="orderbook-cell">${formatQuantity(total)}</td>
        `;

        // Add click handler for quick order entry
        row.addEventListener("click", () => {
            elements.price.value = price;
            elements.side.value = side === "asks" ? "buy" : "sell";
        });

        tableBody.appendChild(row);
    }

    function calculateRunningTotal(levels, targetPrice, isAsk) {
        let total = 0;
        for (const [price, size] of levels) {
            if (isAsk ? parseFloat(price) <= parseFloat(targetPrice) : parseFloat(price) >= parseFloat(targetPrice)) {
                total += parseFloat(size);
            }
            if (parseFloat(price) === parseFloat(targetPrice)) break;
        }
        return total;
    }

    function updateBBODisplay(data) {
        if (data.bids && data.bids.length > 0) {
            elements.bestBid.textContent = formatPrice(data.bids[0][0]);
        } else {
            elements.bestBid.textContent = "--";
        }

        if (data.asks && data.asks.length > 0) {
            elements.bestAsk.textContent = formatPrice(data.asks[0][0]);
        } else {
            elements.bestAsk.textContent = "--";
        }
    }

    function updateStatistics() {
        const { bids, asks } = state.orderBook;
        
        if (bids.length > 0 && asks.length > 0) {
            const bestBid = parseFloat(bids[0][0]);
            const bestAsk = parseFloat(asks[0][0]);
            
            state.statistics.spread = bestAsk - bestBid;
            state.statistics.midPrice = (bestBid + bestAsk) / 2;
            
            elements.spread.textContent = formatPrice(state.statistics.spread);
            elements.midPrice.textContent = formatPrice(state.statistics.midPrice);
        } else {
            elements.spread.textContent = "--";
            elements.midPrice.textContent = "--";
        }
    }

    function addTradeToFeed(trade) {
        const tradeElement = document.createElement("div");
        tradeElement.className = "trade-item new-trade";
        
        const aggressorClass = trade.aggressor_side === "buy" ? "aggressor-buy" : "aggressor-sell";
        const tradeTime = new Date(trade.timestamp * 1000).toLocaleTimeString();
        
        tradeElement.innerHTML = `
            <div>
                <span class="trade-symbol">${trade.symbol}</span>
                <span class="trade-price ${aggressorClass}">${formatPrice(trade.price)}</span>
            </div>
            <div class="trade-quantity">
                <strong>${formatQuantity(trade.quantity)}</strong>
                <span style="margin-left: 0.5rem; color: #64748b;">
                    ${trade.aggressor_side.toUpperCase()}
                </span>
            </div>
            <div class="trade-meta">
                <span>ID: ${trade.trade_id.substring(0, 12)}...</span>
                <span>${tradeTime}</span>
            </div>
        `;

        // Add to beginning of feed
        if (elements.tradeFeed.children.length === 0 || 
            elements.tradeFeed.children[0].textContent.includes("Waiting for trades")) {
            elements.tradeFeed.innerHTML = "";
        }

        elements.tradeFeed.insertBefore(tradeElement, elements.tradeFeed.firstChild);

        // Limit trade history in DOM
        const tradeItems = elements.tradeFeed.children;
        if (tradeItems.length > CONFIG.MAX_TRADE_HISTORY) {
            for (let i = CONFIG.MAX_TRADE_HISTORY; i < tradeItems.length; i++) {
                tradeItems[i].remove();
            }
        }
    }

    // Order management
    async function submitOrder() {
        if (!state.connected) {
            showAlert("Not connected to server", "error");
            return;
        }

        const orderData = {
            symbol: elements.symbol.value || state.currentSymbol,
            side: elements.side.value,
            order_type: elements.orderType.value,
            quantity: elements.quantity.value
        };

        // Add price for limit orders
        if (elements.orderType.value === "limit") {
            if (!elements.price.value) {
                showAlert("Price is required for limit orders", "error");
                return;
            }
            orderData.price = elements.price.value;
        }

        // Add client order ID if provided
        if (elements.clientOrderId.value) {
            orderData.client_order_id = elements.clientOrderId.value;
        }

        try {
            setLoading(elements.submitBtn, true);
            showAlert("Submitting order...", "info");

            const response = await fetch(`${CONFIG.API_HOST}/order`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(orderData)
            });

            const result = await response.json();

            if (response.ok) {
                showAlert(`Order submitted: ${result.order_id}`, "success");
                state.orderCount++;
                elements.orderCount.textContent = state.orderCount;

                // Clear form
                elements.quantity.value = "";
                elements.clientOrderId.value = "";
                
                // Show trades if any
                if (result.trades && result.trades.length > 0) {
                    result.trades.forEach(trade => onTrade(trade));
                }

                showToast(`Order ${result.status}: ${result.order_id.substring(0, 8)}...`, "success");
            } else {
                showAlert(`Order failed: ${result.error}`, "error");
                showToast("Order submission failed", "error");
            }

        } catch (error) {
            console.error("Order submission error:", error);
            showAlert(`Network error: ${error.message}`, "error");
            showToast("Network error occurred", "error");
        } finally {
            setLoading(elements.submitBtn, false);
        }
    }

    async function cancelOrder() {
        const orderId = elements.cancelOrderId.value.trim();
        if (!orderId) {
            showAlert("Please enter an Order ID", "error");
            return;
        }

        try {
            setLoading(elements.cancelBtn, true);

            const response = await fetch(`${CONFIG.API_HOST}/cancel`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    symbol: state.currentSymbol,
                    order_id: orderId
                })
            });

            const result = await response.json();

            if (response.ok && result.success) {
                showAlert(`Order cancelled: ${orderId}`, "success");
                elements.cancelOrderId.value = "";
                showToast("Order cancelled successfully", "success");
            } else {
                showAlert(`Cancel failed: ${result.error || "Unknown error"}`, "error");
                showToast("Order cancellation failed", "error");
            }

        } catch (error) {
            console.error("Cancel error:", error);
            showAlert(`Network error: ${error.message}`, "error");
        } finally {
            setLoading(elements.cancelBtn, false);
        }
    }

    // Event listeners setup
    function setupEventListeners() {
        // Order submission
        elements.submitBtn.addEventListener("click", submitOrder);
        elements.cancelBtn.addEventListener("click", cancelOrder);

        // Symbol change
        elements.symbol.addEventListener("change", (e) => {
            const oldSymbol = state.currentSymbol;
            state.currentSymbol = e.target.value;
            
            // Update subscriptions
            if (state.connected) {
                unsubscribeFromSymbol(oldSymbol);
                subscribeToSymbol(state.currentSymbol);
            }
            
            updateSymbolDisplay();
            clearOrderBook();
            clearTrades();
        });

        // Order type change
        elements.orderType.addEventListener("change", (e) => {
            const isLimit = e.target.value === "limit";
            elements.price.disabled = !isLimit;
            if (!isLimit) {
                elements.price.value = "";
            }
        });

        // Clear trades button
        elements.clearTrades.addEventListener("click", clearTrades);

        // Keyboard shortcuts
        document.addEventListener("keydown", (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case "Enter":
                        e.preventDefault();
                        submitOrder();
                        break;
                    case "Escape":
                        clearForm();
                        break;
                }
            }
        });

        // Form validation
        elements.quantity.addEventListener("input", validateQuantity);
        elements.price.addEventListener("input", validatePrice);
    }

    // Utility functions
    function formatPrice(price) {
        const num = parseFloat(price);
        return num.toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 8
        });
    }

    function formatQuantity(quantity) {
        const num = parseFloat(quantity);
        return num.toLocaleString(undefined, {
            minimumFractionDigits: 0,
            maximumFractionDigits: 8
        });
    }

    function showAlert(message, type) {
        elements.orderAlert.textContent = message;
        elements.orderAlert.className = `alert alert-${type}`;
        elements.orderAlert.style.display = "block";

        // Auto-hide success messages
        if (type === "success") {
            setTimeout(() => {
                elements.orderAlert.style.display = "none";
            }, 3000);
        }
    }

    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `alert alert-${type}`;
        toast.style.cssText = `
            position: relative;
            margin-bottom: 0.5rem;
            animation: slideInRight 0.3s ease;
            max-width: 300px;
        `;
        toast.textContent = message;

        elements.toastContainer.appendChild(toast);

        // Auto-remove toast
        setTimeout(() => {
            toast.style.animation = "slideOutRight 0.3s ease";
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    function setLoading(button, loading) {
        if (loading) {
            button.disabled = true;
            button.classList.add("loading");
            const spinner = document.createElement("span");
            spinner.className = "spinner";
            button.prepend(spinner);
        } else {
            button.disabled = false;
            button.classList.remove("loading");
            const spinner = button.querySelector(".spinner");
            if (spinner) spinner.remove();
        }
    }

    function clearOrderBook() {
        elements.asksTableBody.innerHTML = "";
        elements.bidsTableBody.innerHTML = "";
        elements.bestBid.textContent = "--";
        elements.bestAsk.textContent = "--";
        elements.spread.textContent = "--";
        elements.midPrice.textContent = "--";
    }

    function clearTrades() {
        elements.tradeFeed.innerHTML = `
            <div style="text-align: center; color: #64748b; padding: 2rem;">
                <i class="fas fa-clock"></i>
                <div style="margin-top: 0.5rem;">Waiting for trades...</div>
            </div>
        `;
        state.trades = [];
    }

    function clearForm() {
        elements.quantity.value = "";
        elements.price.value = "";
        elements.clientOrderId.value = "";
        elements.cancelOrderId.value = "";
        elements.orderAlert.style.display = "none";
    }

    function validateQuantity(e) {
        const value = parseFloat(e.target.value);
        if (isNaN(value) || value <= 0) {
            e.target.setCustomValidity("Quantity must be a positive number");
        } else {
            e.target.setCustomValidity("");
        }
    }

    function validatePrice(e) {
        if (elements.orderType.value === "limit") {
            const value = parseFloat(e.target.value);
            if (isNaN(value) || value <= 0) {
                e.target.setCustomValidity("Price must be a positive number");
            } else {
                e.target.setCustomValidity("");
            }
        }
    }

    function startPingTimer() {
        pingInterval = setInterval(() => {
            if (socket && socket.connected) {
                state.lastPingTime = Date.now();
                socket.emit("ping");
            }
        }, CONFIG.PING_INTERVAL);
    }

    function loadSymbols() {
        // Load available symbols from API
        fetch(`${CONFIG.API_HOST}/symbols`)
            .then(response => response.json())
            .then(data => {
                if (data.symbols) {
                    updateSymbolOptions(data.symbols);
                }
            })
            .catch(error => {
                console.warn("Failed to load symbols:", error);
            });
    }

    function updateSymbolOptions(symbols) {
        elements.symbol.innerHTML = "";
        symbols.forEach(symbol => {
            const option = document.createElement("option");
            option.value = symbol;
            option.textContent = symbol;
            option.selected = symbol === state.currentSymbol;
            elements.symbol.appendChild(option);
        });
    }

    function playTradeSound() {
        // Optional: play sound notification for trades
        try {
            const audio = new Audio("data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+HyvmAZBjiB0vLLdSEGJ3PN8thzEgEfa7bv3JFBCw0=" );
            audio.volume = 0.1;
            audio.play().catch(() => {}); // Ignore errors
        } catch (e) {
            // Ignore audio errors
        }
    }

    // Initialize on DOM ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

    // Cleanup on page unload
    window.addEventListener("beforeunload", () => {
        if (socket && socket.connected) {
            socket.disconnect();
        }
        if (pingInterval) {
            clearInterval(pingInterval);
        }
    });

    // Add CSS animations dynamically
    const style = document.createElement("style");
    style.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes slideOutRight {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
        
        .orderbook-row:hover {
            cursor: pointer;
            background-color: rgba(102, 126, 234, 0.05) !important;
        }
        
        .form-input:invalid {
            border-color: #ef4444;
        }
        
        .form-input:valid {
            border-color: #10b981;
        }
    `;
    document.head.appendChild(style);

    // Export functions for debugging (development only)
    if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
        window.matchingEngineDebug = {
            state,
            socket,
            submitOrder,
            cancelOrder,
            connectSocket,
            showToast,
            clearTrades
        };
        console.info("Debug functions available at window.matchingEngineDebug");
    }

})();