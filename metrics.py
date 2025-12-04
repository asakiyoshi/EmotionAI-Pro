# metrics.py
import time
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

@dataclass
class PerformanceMetrics:
    """性能指标"""
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    avg_response_time: float = 0.0
    user_state_loads: int = 0
    user_state_saves: int = 0
    llm_calls: int = 0
    emotion_updates: int = 0
    error_count: int = 0

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.response_times: List[float] = []
        self.error_details: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._start_time = time.time()
    
    async def record_cache_hit(self):
        """记录缓存命中"""
        async with self._lock:
            self.metrics.cache_hit_rate = (
                (self.metrics.cache_hit_rate * self.metrics.user_state_loads + 1) /
                (self.metrics.user_state_loads + 1)
            )
            self.metrics.user_state_loads += 1
    
    async def record_cache_miss(self):
        """记录缓存未命中"""
        async with self._lock:
            self.metrics.cache_miss_rate = (
                (self.metrics.cache_miss_rate * self.metrics.user_state_loads + 1) /
                (self.metrics.user_state_loads + 1)
            )
            self.metrics.user_state_loads += 1
    
    async def record_response_time(self, response_time: float):
        """记录响应时间"""
        async with self._lock:
            self.response_times.append(response_time)
            # 只保留最近100个记录
            if len(self.response_times) > 100:
                self.response_times.pop(0)
            self.metrics.avg_response_time = statistics.mean(self.response_times)
    
    async def record_user_state_save(self):
        """记录用户状态保存"""
        async with self._lock:
            self.metrics.user_state_saves += 1
    
    async def record_llm_call(self):
        """记录LLM调用"""
        async with self._lock:
            self.metrics.llm_calls += 1
    
    async def record_emotion_update(self):
        """记录情感更新"""
        async with self._lock:
            self.metrics.emotion_updates += 1
    
    async def record_error(self, error_type: str, details: str = ""):
        """记录错误"""
        async with self._lock:
            self.metrics.error_count += 1
            self.error_details.append({
                'timestamp': time.time(),
                'type': error_type,
                'details': details
            })
            # 只保留最近50个错误
            if len(self.error_details) > 50:
                self.error_details.pop(0)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        async with self._lock:
            uptime = time.time() - self._start_time
            return {
                **asdict(self.metrics),
                'uptime_seconds': uptime,
                'recent_response_times': self.response_times[-10:] if self.response_times else [],
                'recent_errors': self.error_details[-5:] if self.error_details else [],
                'qps': self.metrics.user_state_loads / uptime if uptime > 0 else 0
            }
    
    async def reset_metrics(self):
        """重置指标"""
        async with self._lock:
            self.metrics = PerformanceMetrics()
            self.response_times.clear()
            self.error_details.clear()
            self._start_time = time.time()