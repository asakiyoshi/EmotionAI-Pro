# performance_middleware.py
"""
性能监控中间件
"""
import time
import asyncio
from typing import Callable, Any
from functools import wraps
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

@dataclass
class PerformanceMetrics:
    """性能指标"""
    call_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    errors: int = 0
    last_called: float = 0.0
    
    def record_call(self, duration: float, had_error: bool = False):
        """记录调用"""
        self.call_count += 1
        self.total_time += duration
        self.avg_time = self.total_time / self.call_count
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.last_called = time.time()
        if had_error:
            self.errors += 1
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_history: int = 100):
        self.metrics: dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self.history: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = asyncio.Lock()
    
    def measure(self, name: str = None):
        """测量装饰器"""
        def decorator(func: Callable):
            metric_name = name or func.__name__
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                had_error = False
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception:
                    had_error = True
                    raise
                finally:
                    duration = time.time() - start_time
                    await self.record_metric(metric_name, duration, had_error)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                had_error = False
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception:
                    had_error = True
                    raise
                finally:
                    duration = time.time() - start_time
                    asyncio.create_task(self.record_metric(metric_name, duration, had_error))
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def record_metric(self, name: str, duration: float, had_error: bool = False):
        """记录指标"""
        async with self.lock:
            metric = self.metrics[name]
            metric.record_call(duration, had_error)
            
            # 记录历史
            self.history[name].append({
                'timestamp': time.time(),
                'duration': duration,
                'had_error': had_error
            })
    
    async def get_metrics(self) -> dict:
        """获取所有指标"""
        async with self.lock:
            return {
                name: metric.to_dict()
                for name, metric in self.metrics.items()
            }
    
    async def get_metric_history(self, name: str, limit: int = 50) -> list:
        """获取指标历史"""
        async with self.lock:
            if name not in self.history:
                return []
            
            history = list(self.history[name])
            return history[-limit:] if limit > 0 else history
    
    async def get_summary(self) -> dict:
        """获取摘要"""
        async with self.lock:
            if not self.metrics:
                return {}
            
            all_durations = []
            for history in self.history.values():
                for entry in history:
                    all_durations.append(entry['duration'])
            
            return {
                'total_metrics': len(self.metrics),
                'total_calls': sum(m.call_count for m in self.metrics.values()),
                'total_errors': sum(m.errors for m in self.metrics.values()),
                'avg_duration': statistics.mean(all_durations) if all_durations else 0,
                'p95_duration': statistics.quantiles(all_durations, n=20)[18] if len(all_durations) >= 20 else 0
            }
    
    async def reset_metrics(self, name: str = None):
        """重置指标"""
        async with self.lock:
            if name:
                if name in self.metrics:
                    self.metrics[name] = PerformanceMetrics()
                if name in self.history:
                    self.history[name].clear()
            else:
                self.metrics.clear()
                self.history.clear()

# 全局监控器实例
global_monitor = PerformanceMonitor()

def monitor_performance(name: str = None):
    """性能监控装饰器"""
    return global_monitor.measure(name)