# health.py
import asyncio
import psutil
import time
import logging  
from typing import Dict, Any
from dataclasses import dataclass

# 创建日志记录器
logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    """系统健康状态"""
    status: str
    memory_usage: float
    cpu_usage: float
    disk_usage: float
    active_users: int
    queue_size: int

class HealthChecker:
    """健康检查器"""
    
    def __init__(self, plugin):
        self.plugin = plugin
        self.start_time = time.time()
    
    async def check_health(self) -> SystemHealth:
        """检查系统健康状态"""
        try:
            # 系统资源使用情况
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage('/')
            
            # 插件特定指标
            cache_stats = await self.plugin.cache.get_stats()
            all_states = await self.plugin.user_manager.repository.get_all_user_states()
            active_users = len(all_states)
            
            # 确定状态
            status = "healthy"
            if memory.percent > 90:
                status = "memory_warning"
            elif cpu > 80:
                status = "cpu_warning"
            elif disk.percent > 90:
                status = "disk_warning"
            
            return SystemHealth(
                status=status,
                memory_usage=memory.percent,
                cpu_usage=cpu,
                disk_usage=disk.percent,
                active_users=active_users,
                queue_size=cache_stats['total_entries']
            )
            
        except Exception as e:
            return SystemHealth(
                status="error",
                memory_usage=0,
                cpu_usage=0,
                disk_usage=0,
                active_users=0,
                queue_size=0
            )
    
    async def get_detailed_health_report(self) -> Dict[str, Any]:
        """获取详细健康报告"""
        health = await self.check_health()
        cache_stats = await self.plugin.cache.get_stats()
        metrics = await self.plugin.metrics_collector.get_metrics()
        
        return {
            'system_health': {
                'status': health.status,
                'memory_usage_percent': health.memory_usage,
                'cpu_usage_percent': health.cpu_usage,
                'disk_usage_percent': health.disk_usage,
                'active_users': health.active_users,
                'uptime_hours': (time.time() - self.start_time) / 3600
            },
            'performance_metrics': metrics,
            'cache_stats': cache_stats,
            'timestamp': time.time()
        }