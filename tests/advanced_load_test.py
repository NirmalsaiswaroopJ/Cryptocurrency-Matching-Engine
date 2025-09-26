# advanced_load_test.py
"""
Advanced Load Testing Suite for Cryptocurrency Matching Engine
Supports high concurrency with detailed metrics and monitoring
"""

import asyncio
import aiohttp
import time
import json
import random
import statistics
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
import argparse
import logging
from datetime import datetime, timedelta
import csv
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Individual test result"""
    timestamp: float
    duration_ms: float
    success: bool
    status_code: Optional[int] = None
    error: Optional[str] = None
    response_size: int = 0
    order_id: Optional[str] = None
    trades: int = 0

@dataclass
class MetricsCollector:
    """Comprehensive metrics collection"""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Latency metrics
    response_times: List[float] = field(default_factory=list)
    min_latency: float = float('inf')
    max_latency: float = 0
    
    # Throughput metrics
    requests_per_second: List[float] = field(default_factory=list)
    
    # Business metrics
    orders_submitted: int = 0
    trades_executed: int = 0
    total_volume: float = 0
    
    # Error tracking
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    status_codes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Concurrency metrics
    active_connections: int = 0
    peak_connections: int = 0
    
    def add_result(self, result: TestResult):
        """Add a test result to metrics"""
        self.total_requests += 1
        
        if result.success:
            self.successful_requests += 1
            self.response_times.append(result.duration_ms)
            self.min_latency = min(self.min_latency, result.duration_ms)
            self.max_latency = max(self.max_latency, result.duration_ms)
            
            if result.order_id:
                self.orders_submitted += 1
            
            self.trades_executed += result.trades
        else:
            self.failed_requests += 1
            if result.error:
                self.error_types[result.error] += 1
        
        if result.status_code:
            self.status_codes[result.status_code] += 1
    
    def get_percentiles(self, percentiles: List[float] = None) -> Dict[str, float]:
        """Calculate response time percentiles"""
        if not self.response_times:
            return {}
        
        if percentiles is None:
            percentiles = [50, 75, 90, 95, 99, 99.9]
        
        sorted_times = sorted(self.response_times)
        result = {}
        
        for p in percentiles:
            index = int(len(sorted_times) * p / 100) - 1
            index = max(0, min(index, len(sorted_times) - 1))
            result[f"p{p}"] = sorted_times[index]
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive test summary"""
        duration = (self.end_time or time.time()) - self.start_time
        
        percentiles = self.get_percentiles()
        
        return {
            "test_duration_seconds": round(duration, 2),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round((self.successful_requests / max(self.total_requests, 1)) * 100, 2),
            "requests_per_second": round(self.total_requests / max(duration, 1), 2),
            "orders_submitted": self.orders_submitted,
            "trades_executed": self.trades_executed,
            "average_latency_ms": round(statistics.mean(self.response_times) if self.response_times else 0, 2),
            "min_latency_ms": round(self.min_latency if self.min_latency != float('inf') else 0, 2),
            "max_latency_ms": round(self.max_latency, 2),
            "median_latency_ms": round(statistics.median(self.response_times) if self.response_times else 0, 2),
            "percentiles": {k: round(v, 2) for k, v in percentiles.items()},
            "peak_connections": self.peak_connections,
            "error_types": dict(self.error_types),
            "status_codes": dict(self.status_codes)
        }


class LoadTester:
    """High-performance async load tester"""
    
    def __init__(self, base_url: str = "http://localhost:5000", 
                 max_connections: int = 1000, timeout: int = 30):
        self.base_url = base_url
        self.max_connections = max_connections
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.metrics = MetricsCollector()
        self.semaphore: Optional[asyncio.Semaphore] = None
        self.running = True
        
        # Test data generators
        self.symbols = ["BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT"]
        self.order_types = ["limit", "market", "ioc", "fok"]
        self.sides = ["buy", "sell"]
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout_config = aiohttp.ClientTimeout(total=self.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout_config,
            headers={"Content-Type": "application/json"}
        )
        
        self.semaphore = asyncio.Semaphore(self.max_connections)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def generate_order_data(self) -> Dict[str, Any]:
        """Generate realistic order data"""
        symbol = random.choice(self.symbols)
        side = random.choice(self.sides)
        order_type = random.choice(self.order_types)
        
        # Generate realistic quantities
        quantity = round(random.uniform(0.001, 1.0), 6)
        
        order_data = {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": str(quantity)
        }
        
        # Add price for limit orders
        if order_type in ["limit", "ioc", "fok"]:
            base_prices = {
                "BTC-USDT": 30000,
                "ETH-USDT": 2000,
                "BNB-USDT": 300,
                "SOL-USDT": 100
            }
            
            base_price = base_prices.get(symbol, 30000)
            # Add some price variation
            price_variation = random.uniform(0.95, 1.05)
            price = round(base_price * price_variation, 2)
            order_data["price"] = str(price)
        
        return order_data
    
    async def submit_single_order(self) -> TestResult:
        """Submit a single order and measure performance"""
        start_time = time.time()
        
        async with self.semaphore:
            self.metrics.active_connections += 1
            self.metrics.peak_connections = max(
                self.metrics.peak_connections, 
                self.metrics.active_connections
            )
            
            try:
                order_data = self.generate_order_data()
                
                async with self.session.post(
                    f"{self.base_url}/order",
                    json=order_data
                ) as response:
                    response_text = await response.text()
                    duration_ms = (time.time() - start_time) * 1000
                    
                    result = TestResult(
                        timestamp=start_time,
                        duration_ms=duration_ms,
                        success=response.status == 200,
                        status_code=response.status,
                        response_size=len(response_text)
                    )
                    
                    if response.status == 200:
                        try:
                            response_data = json.loads(response_text)
                            result.order_id = response_data.get("order_id")
                            result.trades = len(response_data.get("trades", []))
                        except json.JSONDecodeError:
                            result.error = "Invalid JSON response"
                            result.success = False
                    else:
                        result.error = f"HTTP {response.status}"
                    
                    return result
                    
            except asyncio.TimeoutError:
                duration_ms = (time.time() - start_time) * 1000
                return TestResult(
                    timestamp=start_time,
                    duration_ms=duration_ms,
                    success=False,
                    error="Timeout"
                )
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                return TestResult(
                    timestamp=start_time,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e)
                )
            finally:
                self.metrics.active_connections -= 1
    
    async def run_constant_rate_test(self, rate_per_second: int, duration_seconds: int):
        """Run test at constant rate for specified duration"""
        logger.info(f"Starting constant rate test: {rate_per_second} req/sec for {duration_seconds}s")
        
        end_time = time.time() + duration_seconds
        interval = 1.0 / rate_per_second
        
        tasks = []
        
        while time.time() < end_time and self.running:
            # Start new request
            task = asyncio.create_task(self.submit_single_order())
            tasks.append(task)
            
            # Process completed tasks
            done_tasks = [t for t in tasks if t.done()]
            for task in done_tasks:
                try:
                    result = await task
                    self.metrics.add_result(result)
                except Exception as e:
                    logger.error(f"Task error: {e}")
                tasks.remove(task)
            
            # Rate limiting
            await asyncio.sleep(interval)
        
        # Wait for remaining tasks
        if tasks:
            logger.info(f"Waiting for {len(tasks)} remaining requests...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, TestResult):
                    self.metrics.add_result(result)
                elif isinstance(result, Exception):
                    logger.error(f"Final task error: {result}")
        
        self.metrics.end_time = time.time()
        logger.info("Constant rate test completed")
    
    async def run_spike_test(self, peak_rate: int, spike_duration: int, 
                           baseline_rate: int, total_duration: int):
        """Run spike test with sudden traffic increase"""
        logger.info(f"Starting spike test: baseline {baseline_rate}/sec, spike to {peak_rate}/sec")
        
        start_time = time.time()
        spike_start = start_time + (total_duration - spike_duration) / 2
        spike_end = spike_start + spike_duration
        
        while time.time() - start_time < total_duration and self.running:
            current_time = time.time()
            
            # Determine current rate
            if spike_start <= current_time <= spike_end:
                current_rate = peak_rate
            else:
                current_rate = baseline_rate
            
            # Submit request
            task = asyncio.create_task(self.submit_single_order())
            result = await task
            self.metrics.add_result(result)
            
            # Rate limiting
            interval = 1.0 / current_rate
            await asyncio.sleep(interval)
        
        self.metrics.end_time = time.time()
        logger.info("Spike test completed")
    
    def stop(self):
        """Stop the test gracefully"""
        self.running = False
        logger.info("Stopping load test...")


class MetricsReporter:
    """Advanced metrics reporting and analysis"""
    
    @staticmethod
    def print_summary(metrics: MetricsCollector):
        """Print comprehensive test summary"""
        summary = metrics.get_summary()
        
        print("\n" + "="*80)
        print("LOAD TEST RESULTS")
        print("="*80)
        
        # Test Overview
        print(f"Test Duration: {summary['test_duration_seconds']}s")
        print(f"Total Requests: {summary['total_requests']:,}")
        print(f"Successful: {summary['successful_requests']:,} ({summary['success_rate']}%)")
        print(f"Failed: {summary['failed_requests']:,}")
        print(f"Requests/Second: {summary['requests_per_second']:,.2f}")
        
        # Business Metrics
        print(f"\nBUSINESS METRICS:")
        print(f"Orders Submitted: {summary['orders_submitted']:,}")
        print(f"Trades Executed: {summary['trades_executed']:,}")
        print(f"Trade Rate: {(summary['trades_executed'] / max(summary['orders_submitted'], 1) * 100):.2f}%")
        
        # Latency Metrics
        print(f"\nLATENCY METRICS (ms):")
        print(f"Average: {summary['average_latency_ms']}")
        print(f"Median: {summary['median_latency_ms']}")
        print(f"Min: {summary['min_latency_ms']}")
        print(f"Max: {summary['max_latency_ms']}")
        
        print("\nPERCENTILES (ms):")
        for percentile, value in summary['percentiles'].items():
            print(f"  {percentile}: {value}")
        
        # Connection Metrics
        print(f"\nCONCURRENCY:")
        print(f"Peak Connections: {summary['peak_connections']}")
        
        # Error Analysis
        if summary['error_types']:
            print(f"\nERROR BREAKDOWN:")
            for error, count in summary['error_types'].items():
                print(f"  {error}: {count}")
        
        if summary['status_codes']:
            print(f"\nSTATUS CODES:")
            for code, count in summary['status_codes'].items():
                print(f"  {code}: {count}")
        
        print("="*80)
    
    @staticmethod
    def save_detailed_report(metrics: MetricsCollector, filename: str = None):
        """Save detailed CSV report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"load_test_report_{timestamp}.csv"
        
        summary = metrics.get_summary()
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write summary
            writer.writerow(["SUMMARY"])
            for key, value in summary.items():
                if isinstance(value, dict):
                    writer.writerow([key, ""])
                    for sub_key, sub_value in value.items():
                        writer.writerow([f"  {sub_key}", sub_value])
                else:
                    writer.writerow([key, value])
        
        logger.info(f"Detailed report saved to {filename}")


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Advanced Load Testing for Matching Engine")
    parser.add_argument("--url", default="http://localhost:5000", help="API base URL")
    parser.add_argument("--rate", type=int, default=100, help="Requests per second")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--connections", type=int, default=100, help="Max concurrent connections")
    parser.add_argument("--test-type", choices=["constant", "spike"], default="constant", help="Test type")
    parser.add_argument("--spike-rate", type=int, default=200, help="Peak rate for spike test")
    parser.add_argument("--spike-duration", type=int, default=30, help="Spike duration in seconds")
    parser.add_argument("--baseline-rate", type=int, default=100, help="Baseline rate for spike test")
    parser.add_argument("--report", help="Save detailed report to file")
    
    args = parser.parse_args()
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received interrupt signal, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    async with LoadTester(args.url, args.connections) as tester:
        try:
            if args.test_type == "constant":
                await tester.run_constant_rate_test(args.rate, args.duration)
            elif args.test_type == "spike":
                await tester.run_spike_test(
                    args.spike_rate, 
                    args.spike_duration,
                    args.baseline_rate, 
                    args.duration
                )
            
            # Print results
            MetricsReporter.print_summary(tester.metrics)
            
            # Save detailed report if requested
            if args.report:
                MetricsReporter.save_detailed_report(tester.metrics, args.report)
        
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            tester.stop()
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())