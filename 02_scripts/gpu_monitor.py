"""
GPU Monitoring and Performance Utilities for RTX 3070
Real-time monitoring and optimization suggestions
"""

import torch
import psutil
import time
import json
from pathlib import Path
from datetime import datetime
import subprocess
import sys

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("[WARNING] pynvml not installed. Install with: pip install nvidia-ml-py3")


class GPUMonitor:
    """
    Real-time GPU and system monitoring
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if NVML_AVAILABLE:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        else:
            self.handle = None

    def get_gpu_stats(self):
        """Get current GPU statistics"""
        if not torch.cuda.is_available():
            return None

        stats = {
            'name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
        }

        # Memory stats
        stats['memory_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
        stats['memory_reserved_gb'] = torch.cuda.memory_reserved() / 1024**3
        stats['memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        stats['memory_free_gb'] = stats['memory_total_gb'] - stats['memory_allocated_gb']
        stats['memory_utilization_percent'] = (stats['memory_allocated_gb'] / stats['memory_total_gb']) * 100

        # NVML stats (if available)
        if NVML_AVAILABLE and self.handle:
            try:
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                stats['gpu_utilization_percent'] = util.gpu
                stats['memory_controller_utilization_percent'] = util.memory

                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                stats['temperature_c'] = temp

                # Power
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to watts
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0
                stats['power_usage_w'] = power
                stats['power_limit_w'] = power_limit
                stats['power_utilization_percent'] = (power / power_limit) * 100

                # Clock speeds
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
                memory_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
                stats['graphics_clock_mhz'] = graphics_clock
                stats['memory_clock_mhz'] = memory_clock

            except Exception as e:
                print(f"[WARNING] Could not get NVML stats: {e}")

        return stats

    def get_cpu_stats(self):
        """Get CPU statistics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        }

    def get_ram_stats(self):
        """Get RAM statistics"""
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / 1024**3,
            'available_gb': mem.available / 1024**3,
            'used_gb': mem.used / 1024**3,
            'percent': mem.percent,
        }

    def print_stats(self, clear_screen=False):
        """Print formatted statistics"""
        if clear_screen:
            print("\033[H\033[J", end="")  # Clear screen

        print("="*80)
        print(f"GPU MONITORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # GPU Stats
        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            print("\n📊 GPU (RTX 3070):")
            print(f"  Name: {gpu_stats['name']}")
            print(f"  CUDA Version: {gpu_stats['cuda_version']}")
            print(f"  PyTorch Version: {gpu_stats['pytorch_version']}")

            print(f"\n  💾 VRAM:")
            print(f"    Allocated: {gpu_stats['memory_allocated_gb']:.2f} GB / {gpu_stats['memory_total_gb']:.2f} GB")
            print(f"    Free: {gpu_stats['memory_free_gb']:.2f} GB")
            print(f"    Utilization: {gpu_stats['memory_utilization_percent']:.1f}%")
            print(f"    {'  [' + '█' * int(gpu_stats['memory_utilization_percent']/5) + '░' * (20-int(gpu_stats['memory_utilization_percent']/5)) + ']'}")

            if 'gpu_utilization_percent' in gpu_stats:
                print(f"\n  ⚡ GPU Utilization: {gpu_stats['gpu_utilization_percent']}%")
                print(f"    {'  [' + '█' * int(gpu_stats['gpu_utilization_percent']/5) + '░' * (20-int(gpu_stats['gpu_utilization_percent']/5)) + ']'}")

            if 'temperature_c' in gpu_stats:
                print(f"\n  🌡️  Temperature: {gpu_stats['temperature_c']}°C")

            if 'power_usage_w' in gpu_stats:
                print(f"\n  🔋 Power:")
                print(f"    Usage: {gpu_stats['power_usage_w']:.1f}W / {gpu_stats['power_limit_w']:.1f}W")
                print(f"    Utilization: {gpu_stats['power_utilization_percent']:.1f}%")

            if 'graphics_clock_mhz' in gpu_stats:
                print(f"\n  ⚙️  Clock Speeds:")
                print(f"    Graphics: {gpu_stats['graphics_clock_mhz']} MHz")
                print(f"    Memory: {gpu_stats['memory_clock_mhz']} MHz")

        # CPU Stats
        cpu_stats = self.get_cpu_stats()
        print(f"\n🖥️  CPU (Ryzen 7 5800X):")
        print(f"  Cores: {cpu_stats['cpu_count']}")
        print(f"  Usage: {cpu_stats['cpu_percent']:.1f}%")
        if cpu_stats['cpu_freq_mhz'] > 0:
            print(f"  Frequency: {cpu_stats['cpu_freq_mhz']:.0f} MHz")

        # RAM Stats
        ram_stats = self.get_ram_stats()
        print(f"\n💿 RAM:")
        print(f"  Total: {ram_stats['total_gb']:.1f} GB")
        print(f"  Used: {ram_stats['used_gb']:.1f} GB / Available: {ram_stats['available_gb']:.1f} GB")
        print(f"  Utilization: {ram_stats['percent']:.1f}%")
        print(f"  {'[' + '█' * int(ram_stats['percent']/5) + '░' * (20-int(ram_stats['percent']/5)) + ']'}")

        print("\n" + "="*80)

        return {
            'gpu': gpu_stats,
            'cpu': cpu_stats,
            'ram': ram_stats
        }

    def monitor_continuous(self, interval=2):
        """Continuous monitoring"""
        print("Starting continuous monitoring (Press Ctrl+C to stop)...\n")

        try:
            while True:
                self.print_stats(clear_screen=True)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")

    def check_optimization(self):
        """Check system and provide optimization suggestions"""
        print("="*80)
        print("SYSTEM OPTIMIZATION CHECK")
        print("="*80)

        stats = self.print_stats()

        print("\n" + "="*80)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("="*80)

        recommendations = []

        # Check GPU
        if stats['gpu']:
            if stats['gpu']['memory_utilization_percent'] > 90:
                recommendations.append("⚠️  VRAM usage >90%. Reduce batch size or model size.")
            elif stats['gpu']['memory_utilization_percent'] < 50:
                recommendations.append("✅ VRAM usage <50%. You can increase batch size for better performance.")

            if 'gpu_utilization_percent' in stats['gpu']:
                if stats['gpu']['gpu_utilization_percent'] < 70:
                    recommendations.append("⚠️  GPU utilization <70%. Check if CPU/data loading is bottleneck.")
                else:
                    recommendations.append("✅ Good GPU utilization!")

            if 'temperature_c' in stats['gpu']:
                if stats['gpu']['temperature_c'] > 80:
                    recommendations.append("⚠️  GPU temperature >80°C. Check cooling.")
                else:
                    recommendations.append("✅ GPU temperature is good.")

        # Check CPU
        if stats['cpu']['cpu_percent'] > 90:
            recommendations.append("⚠️  CPU usage >90%. May bottleneck GPU. Reduce DataLoader workers.")
        elif stats['cpu']['cpu_percent'] < 30:
            recommendations.append("✅ CPU usage is low. Can increase DataLoader workers.")

        # Check RAM
        if stats['ram']['percent'] > 90:
            recommendations.append("⚠️  RAM usage >90%. Reduce RAM caching or batch size.")
        else:
            recommendations.append("✅ RAM usage is healthy.")

        # Print recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec}")
        else:
            print("\n✅ No issues detected. System is optimized!")

        print("\n" + "="*80)

        return recommendations

    def benchmark_throughput(self, batch_size=16, num_batches=10, image_size=(3, 512, 512)):
        """Benchmark GPU throughput"""
        print("="*80)
        print("GPU THROUGHPUT BENCHMARK")
        print("="*80)
        print(f"\nBatch size: {batch_size}")
        print(f"Image size: {image_size}")
        print(f"Number of batches: {num_batches}")

        if not torch.cuda.is_available():
            print("\n[ERROR] CUDA not available!")
            return

        # Warmup
        print("\nWarming up...")
        dummy_input = torch.randn(batch_size, *image_size, device=self.device)
        for _ in range(5):
            _ = dummy_input * 2

        torch.cuda.synchronize()

        # Benchmark
        print("Running benchmark...")
        start_time = time.time()

        for _ in range(num_batches):
            dummy_input = torch.randn(batch_size, *image_size, device=self.device)
            with torch.cuda.amp.autocast():
                result = dummy_input * 2 + 1
            torch.cuda.synchronize()

        end_time = time.time()
        duration = end_time - start_time

        # Results
        total_images = batch_size * num_batches
        throughput = total_images / duration

        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"\nTotal images: {total_images}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Throughput: {throughput:.2f} images/second")
        print(f"Time per image: {1000 * duration / total_images:.2f} ms")

        # GPU stats after benchmark
        print("\n" + "="*80)
        self.print_stats()

        return throughput

    def __del__(self):
        """Cleanup"""
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


def main():
    """Main monitoring interface"""
    monitor = GPUMonitor()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'watch':
            # Continuous monitoring
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 2
            monitor.monitor_continuous(interval)

        elif command == 'check':
            # Optimization check
            monitor.check_optimization()

        elif command == 'benchmark':
            # Throughput benchmark
            batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
            monitor.benchmark_throughput(batch_size=batch_size)

        else:
            print(f"Unknown command: {command}")
            print_help()

    else:
        # Single snapshot
        monitor.print_stats()
        print("\nFor continuous monitoring: python gpu_monitor.py watch")
        print("For optimization check: python gpu_monitor.py check")
        print("For benchmark: python gpu_monitor.py benchmark [batch_size]")


def print_help():
    print("""
GPU Monitor - Usage:

  python gpu_monitor.py              - Show current stats
  python gpu_monitor.py watch [sec]  - Continuous monitoring (default: 2 sec)
  python gpu_monitor.py check        - Check optimization status
  python gpu_monitor.py benchmark    - Benchmark GPU throughput

Examples:
  python gpu_monitor.py watch 1      - Monitor every 1 second
  python gpu_monitor.py benchmark 32 - Benchmark with batch size 32
""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
