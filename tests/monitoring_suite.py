# monitoring_suite.py
"""
Real-time monitoring and profiling tools for the matching engine
"""

import psutil
import time
import threading
import json
import requests
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from datetime import datetime
import numpy as np

class SystemMonitor:
    """Monitor system resources during load testing"""
    
    def __init__(self, api_url: str = "http://localhost:5000", interval: float = 1.0):
        self.api_url = api_url
        self.interval = interval
        self.running = False
        
        # Metrics storage
        self.timestamps = deque(maxlen=1000)
        self.cpu_usage = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.network_io = deque(maxlen=1000)
        self.disk_io = deque(maxlen=1000)
        
        # API metrics
        self.response_times = deque(maxlen=1000)
        self.active_connections = deque(maxlen=1000)
        self.orders_per_sec = deque(maxlen=1000)
        self.trades_per_sec = deque(maxlen=1000)
        
        # Process monitoring
        self.process = None
        self.find_matching_engine_process()
    
    def find_matching_engine_process(self):
        """Find the matching engine process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'app.py' in cmdline or 'matching' in cmdline:
                        self.process = psutil.Process(proc.info['pid'])
                        print(f"Found matching engine process: PID {proc.info['pid']}")
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    
    def start_monitoring(self):
        """Start monitoring in background thread"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("System monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        print("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        prev_net_io = psutil.net_io_counters()
        prev_disk_io = psutil.disk_io_counters()
        
        while self.running:
            timestamp = time.time()
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Network I/O
            net_io = psutil.net_io_counters()
            net_speed = ((net_io.bytes_sent + net_io.bytes_recv) - 
                        (prev_net_io.bytes_sent + prev_net_io.bytes_recv)) / self.interval / 1024 / 1024  # MB/s
            prev_net_io = net_io
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io and prev_disk_io:
                disk_speed = ((disk_io.read_bytes + disk_io.write_bytes) - 
                             (prev_disk_io.read_bytes + prev_disk_io.write_bytes)) / self.interval / 1024 / 1024  # MB/s
                prev_disk_io = disk_io
            else:
                disk_speed = 0
            
            # API health check
            api_response_time = self._check_api_health()
            
            # Store metrics
            self.timestamps.append(timestamp)
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory.percent)
            self.network_io.append(net_speed)
            self.disk_io.append(disk_speed)
            self.response_times.append(api_response_time)
            
            time.sleep(self.interval)
    
    def _check_api_health(self) -> float:
        """Check API response time"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_url}/health", timeout=5)
            response_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                data = response.json()
                # Extract metrics if available
                return response_time
            return response_time
        except Exception:
            return -1  # Indicate failure
    
    def get_current_stats(self) -> dict:
        """Get current system statistics"""
        if not self.timestamps:
            return {}
        
        return {
            "timestamp": self.timestamps[-1],
            "cpu_percent": self.cpu_usage[-1],
            "memory_percent": self.memory_usage[-1],
            "network_io_mbps": self.network_io[-1],
            "disk_io_mbps": self.disk_io[-1],
            "api_response_time_ms": self.response_times[-1],
            "process_info": self._get_process_info()
        }
    
    def _get_process_info(self) -> dict:
        """Get process-specific information"""
        if not self.process:
            return {}
        
        try:
            return {
                "cpu_percent": self.process.cpu_percent(),
                "memory_mb": self.process.memory_info().rss / 1024 / 1024,
                "num_threads": self.process.num_threads(),
                "num_fds": self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
                "connections": len(self.process.connections()) if hasattr(self.process, 'connections') else 0
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}
    
    def plot_metrics(self, save_path: str = None):
        """Plot system metrics"""
        if len(self.timestamps) < 2:
            print("Not enough data to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('System Performance Metrics')
        
        times = [(t - self.timestamps[0]) for t in self.timestamps]
        
        # CPU Usage
        axes[0, 0].plot(times, list(self.cpu_usage))
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_ylabel('Percent')
        
        # Memory Usage
        axes[0, 1].plot(times, list(self.memory_usage))
        axes[0, 1].set_title('Memory Usage (%)')
        axes[0, 1].set_ylabel('Percent')
        
        # Network I/O
        axes[0, 2].plot(times, list(self.network_io))
        axes[0, 2].set_title('Network I/O (MB/s)')
        axes[0, 2].set_ylabel('MB/s')
        
        # Disk I/O
        axes[1, 0].plot(times, list(self.disk_io))
        axes[1, 0].set_title('Disk I/O (MB/s)')
        axes[1, 0].set_ylabel('MB/s')
        
        # API Response Time
        valid_response_times = [rt for rt in self.response_times if rt > 0]
        if valid_response_times:
            axes[1, 1].plot(times[-len(valid_response_times):], valid_response_times)
            axes[1, 1].set_title('API Response Time (ms)')
            axes[1, 1].set_ylabel('Milliseconds')
        
        # Resource Utilization Summary
        if self.process:
            proc_info = self._get_process_info()
            axes[1, 2].bar(['CPU %', 'Memory MB', 'Threads', 'Connections'], 
                          [proc_info.get('cpu_percent', 0),
                           proc_info.get('memory_mb', 0),
                           proc_info.get('num_threads', 0),
                           proc_info.get('connections', 0)])
            axes[1, 2].set_title('Process Statistics')
        
        for ax in axes.flat:
            ax.set_xlabel('Time (seconds)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to {save_path}")
        
        plt.show()


class LoadTestOrchestrator:
    """Orchestrate multiple load tests with monitoring"""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self.monitor = SystemMonitor(api_url)
        self.test_results = []
    
    async def run_comprehensive_test_suite(self):
        """Run a comprehensive suite of load tests"""
        print("Starting comprehensive load test suite...")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        test_scenarios = [
            ("Warmup", {"rate": 10, "duration": 30}),
            ("Low Load", {"rate": 100, "duration": 60}),
            ("Medium Load", {"rate": 500, "duration": 120}),
            ("High Load", {"rate": 1000, "duration": 180}),
            ("Spike Test", {"rate": 100, "duration": 300, "test_type": "spike", 
                          "spike_rate": 2000, "spike_duration": 60}),
        ]
        
        try:
            from advanced_load_test import LoadTester
            
            for scenario_name, config in test_scenarios:
                print(f"\n{'='*50}")
                print(f"Starting: {scenario_name}")
                print(f"{'='*50}")
                
                # Wait between tests
                if self.test_results:
                    print("Cooling down for 30 seconds...")
                    time.sleep(30)
                
                async with LoadTester(self.api_url, config.get("connections", 1000)) as tester:
                    if config.get("test_type") == "spike":
                        await tester.run_spike_test(
                            config["spike_rate"],
                            config["spike_duration"],
                            config["rate"],
                            config["duration"]
                        )
                    else:
                        await tester.run_constant_rate_test(config["rate"], config["duration"])
                    
                    # Store results
                    self.test_results.append({
                        "scenario": scenario_name,
                        "config": config,
                        "metrics": tester.metrics.get_summary(),
                        "system_stats": self.monitor.get_current_stats()
                    })
                
                print(f"{scenario_name} completed")
        
        finally:
            # Stop monitoring
            self.monitor.stop_monitoring()
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"comprehensive_load_test_report_{timestamp}.json"
        
        report = {
            "test_suite": "Comprehensive Load Test",
            "timestamp": timestamp,
            "api_url": self.api_url,
            "scenarios": self.test_results,
            "summary": self._calculate_summary_stats()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nComprehensive report saved to: {report_file}")
        
        # Generate plots
        plot_file = f"performance_metrics_{timestamp}.png"
        self.monitor.plot_metrics(plot_file)
        
        # Print summary
        self._print_summary_report()
    
    def _calculate_summary_stats(self) -> dict:
        """Calculate summary statistics across all tests"""
        if not self.test_results:
            return {}
        
        all_latencies = []
        total_requests = 0
        total_successful = 0
        max_rps = 0
        
        for result in self.test_results:
            metrics = result["metrics"]
            total_requests += metrics.get("total_requests", 0)
            total_successful += metrics.get("successful_requests", 0)
            max_rps = max(max_rps, metrics.get("requests_per_second", 0))
            
            # Collect latency data
            if "percentiles" in metrics:
                all_latencies.extend([metrics["average_latency_ms"]])
        
        return {
            "total_requests_across_all_tests": total_requests,
            "overall_success_rate": (total_successful / max(total_requests, 1)) * 100,
            "peak_requests_per_second": max_rps,
            "average_latency_across_tests": sum(all_latencies) / len(all_latencies) if all_latencies else 0
        }
    
    def _print_summary_report(self):
        """Print summary of all test results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE LOAD TEST SUMMARY")
        print("="*80)
        
        for i, result in enumerate(self.test_results):
            scenario = result["scenario"]
            metrics = result["metrics"]
            
            print(f"\n{i+1}. {scenario}")
            print("-" * (len(scenario) + 3))
            print(f"Requests/sec: {metrics.get('requests_per_second', 0):,.2f}")
            print(f"Success Rate: {metrics.get('success_rate', 0):.2f}%")
            print(f"Avg Latency: {metrics.get('average_latency_ms', 0):.2f}ms")
            print(f"P99 Latency: {metrics.get('percentiles', {}).get('p99', 0):.2f}ms")
            print(f"Orders: {metrics.get('orders_submitted', 0):,}")
            print(f"Trades: {metrics.get('trades_executed', 0):,}")
        
        summary = self._calculate_summary_stats()
        print(f"\nOVERALL SUMMARY:")
        print(f"Total Requests: {summary.get('total_requests_across_all_tests', 0):,}")
        print(f"Overall Success Rate: {summary.get('overall_success_rate', 0):.2f}%")
        print(f"Peak RPS: {summary.get('peak_requests_per_second', 0):,.2f}")
        print("="*80)


# Simple standalone scripts for quick testing

def quick_burst_test(url: str = "http://localhost:5000", concurrent: int = 100, requests: int = 1000):
    """Quick burst test using threading"""
    import threading
    import requests
    import time
    from collections import deque
    
    results = deque()
    
    def make_request():
        try:
            start = time.time()
            response = requests.post(
                f"{url}/order",
                json={
                    "symbol": "BTC-USDT",
                    "side": "buy",
                    "order_type": "limit",
                    "quantity": "0.001",
                    "price": "29000"
                },
                timeout=10
            )
            duration = (time.time() - start) * 1000
            results.append({
                "success": response.status_code == 200,
                "duration_ms": duration,
                "status": response.status_code
            })
        except Exception as e:
            results.append({
                "success": False,
                "duration_ms": -1,
                "error": str(e)
            })
    
    print(f"Starting burst test: {requests} requests with {concurrent} concurrent threads")
    start_time = time.time()
    
    # Create and start threads
    threads = []
    requests_per_thread = requests // concurrent
    
    for _ in range(concurrent):
        for _ in range(requests_per_thread):
            t = threading.Thread(target=make_request)
            threads.append(t)
    
    # Start all threads
    for t in threads:
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    valid_durations = [r["duration_ms"] for r in results if r["duration_ms"] > 0]
    
    print(f"\nBURST TEST RESULTS:")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Requests: {len(results)}")
    print(f"Successful: {successful} ({successful/len(results)*100:.2f}%)")
    print(f"Failed: {failed}")
    print(f"RPS: {len(results)/total_time:.2f}")
    
    if valid_durations:
        print(f"Avg Latency: {sum(valid_durations)/len(valid_durations):.2f}ms")
        print(f"Min Latency: {min(valid_durations):.2f}ms")
        print(f"Max Latency: {max(valid_durations):.2f}ms")


def profile_single_request(url: str = "http://localhost:5000", iterations: int = 100):
    """Profile single request performance"""
    import requests
    import time
    import statistics
    
    durations = []
    
    print(f"Profiling single requests ({iterations} iterations)...")
    
    for i in range(iterations):
        try:
            start = time.time()
            response = requests.post(
                f"{url}/order",
                json={
                    "symbol": "BTC-USDT",
                    "side": "buy",
                    "order_type": "limit",
                    "quantity": "0.001",
                    "price": str(29000 + i)  # Vary price slightly
                },
                timeout=5
            )
            duration = (time.time() - start) * 1000
            
            if response.status_code == 200:
                durations.append(duration)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{iterations} requests...")
                
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    if durations:
        print(f"\nSINGLE REQUEST PROFILE:")
        print(f"Successful requests: {len(durations)}")
        print(f"Average latency: {statistics.mean(durations):.2f}ms")
        print(f"Median latency: {statistics.median(durations):.2f}ms")
        print(f"Min latency: {min(durations):.2f}ms")
        print(f"Max latency: {max(durations):.2f}ms")
        print(f"Std deviation: {statistics.stdev(durations):.2f}ms")


if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Monitoring and Testing Suite")
    parser.add_argument("--url", default="http://localhost:5000", help="API URL")
    parser.add_argument("--test", choices=["monitor", "comprehensive", "burst", "profile"], 
                       default="monitor", help="Test type to run")
    parser.add_argument("--concurrent", type=int, default=100, help="Concurrent requests for burst test")
    parser.add_argument("--requests", type=int, default=1000, help="Total requests for burst test")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations for profile test")
    
    args = parser.parse_args()
    
    if args.test == "monitor":
        monitor = SystemMonitor(args.url)
        try:
            monitor.start_monitoring()
            print("Monitoring system... Press Ctrl+C to stop and generate report")
            while True:
                stats = monitor.get_current_stats()
                if stats:
                    print(f"CPU: {stats.get('cpu_percent', 0):.1f}% | "
                          f"Memory: {stats.get('memory_percent', 0):.1f}% | "
                          f"API: {stats.get('api_response_time_ms', -1):.1f}ms")
                time.sleep(5)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            monitor.plot_metrics()
    
    elif args.test == "comprehensive":
        orchestrator = LoadTestOrchestrator(args.url)
        asyncio.run(orchestrator.run_comprehensive_test_suite())
    
    elif args.test == "burst":
        quick_burst_test(args.url, args.concurrent, args.requests)
    
    elif args.test == "profile":
        profile_single_request(args.url, args.iterations)