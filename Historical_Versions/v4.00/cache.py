# cache.py
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict
import hashlib
import xxhash  # 添加更快的哈希库
from dataclasses import dataclass
from contextlib import asynccontextmanager
import sys

from .constants import CacheConstants, TimeConstants
from .models import CacheStats

# 尝试导入xxhash，如果失败则使用内置哈希
try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    print("警告: xxhash未安装，使用内置哈希函数")

class LRUCacheShard:
    """LRU缓存分片 - 增强版本"""
    
    def __init__(self, max_size: int = 100, shard_id: int = 0):
        self.max_size = max_size
        self.shard_id = shard_id
        self.cache: OrderedDict[str, Tuple[Any, float, float]] = OrderedDict()  # (value, expires_at, access_time)
        self.stats = CacheStats()
        self.lock = asyncio.Lock()
        self.total_size = 0  # 跟踪总大小（字节）
        self.max_memory_bytes = 10 * 1024 * 1024  # 10MB 每个分片限制
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        async with self.lock:
            self.stats.access_count += 1
            
            if key not in self.cache:
                return None
                
            value, expires_at, _ = self.cache[key]
            current_time = time.time()
            
            if current_time >= expires_at:
                # 过期删除
                del self.cache[key]
                self._update_size(key, value, remove=True)
                return None
            
            # 更新访问时间和移动到最近使用
            self.cache[key] = (value, expires_at, current_time)
            self.cache.move_to_end(key)
            self.stats.hit_count += 1
            return value
    
    async def set(self, key: str, value: Any, ttl: int = CacheConstants.DEFAULT_TTL):
        """设置缓存值"""
        async with self.lock:
            current_time = time.time()
            expires_at = current_time + ttl
            
            # 如果键已存在，先删除旧值
            if key in self.cache:
                old_value, _, _ = self.cache[key]
                self._update_size(key, old_value, remove=True)
                del self.cache[key]
            
            # 检查大小限制（条目数和内存）
            if len(self.cache) >= self.max_size:
                await self._evict_oldest()
            
            # 检查内存限制
            estimated_size = self._estimate_size(key, value)
            if self.total_size + estimated_size > self.max_memory_bytes:
                await self._evict_by_memory()
            
            self.cache[key] = (value, expires_at, current_time)
            self._update_size(key, value, remove=False)
    
    def _estimate_size(self, key: str, value: Any) -> int:
        """估算对象大小"""
        try:
            # 简单估算：字符串长度 + 固定开销
            size = len(key.encode('utf-8'))
            if isinstance(value, str):
                size += len(value.encode('utf-8'))
            elif isinstance(value, (int, float, bool)):
                size += 8
            elif isinstance(value, dict):
                size += sum(len(str(k)) + len(str(v)) for k, v in value.items()) * 2
            elif isinstance(value, list):
                size += sum(len(str(item)) for item in value) * 2
            else:
                size += sys.getsizeof(value) if hasattr(sys, 'getsizeof') else 100
            
            return max(100, size)  # 最小100字节
        except:
            return 1000  # 默认1KB
    
    def _update_size(self, key: str, value: Any, remove: bool = False):
        """更新大小跟踪"""
        estimated_size = self._estimate_size(key, value)
        if remove:
            self.total_size = max(0, self.total_size - estimated_size)
        else:
            self.total_size += estimated_size
    
    async def _evict_oldest(self, count: int = 1):
        """淘汰最旧的项目"""
        evicted = 0
        while evicted < count and self.cache:
            oldest_key, (oldest_value, _, _) = self.cache.popitem(last=False)
            self._update_size(oldest_key, oldest_value, remove=True)
            self.stats.eviction_count += 1
            evicted += 1
    
    async def _evict_by_memory(self):
        """根据内存使用淘汰项目"""
        target_reduction = self.total_size - (self.max_memory_bytes * 0.8)  # 减少到80%
        bytes_freed = 0
        
        while bytes_freed < target_reduction and self.cache:
            oldest_key, (oldest_value, _, _) = self.cache.popitem(last=False)
            item_size = self._estimate_size(oldest_key, oldest_value)
            bytes_freed += item_size
            self.total_size -= item_size
            self.stats.eviction_count += 1
    
    async def delete(self, key: str) -> bool:
        """删除缓存键"""
        async with self.lock:
            if key in self.cache:
                value, _, _ = self.cache[key]
                del self.cache[key]
                self._update_size(key, value, remove=True)
                return True
            return False
    
    async def clear(self):
        """清空缓存"""
        async with self.lock:
            self.cache.clear()
            self.total_size = 0
    
    async def cleanup_expired(self) -> Tuple[int, int]:
        """清理过期条目，返回(清理数量, 释放字节)"""
        async with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, (value, expires_at, _) in self.cache.items():
                if current_time >= expires_at:
                    expired_keys.append((key, value))
            
            bytes_freed = 0
            for key, value in expired_keys:
                del self.cache[key]
                item_size = self._estimate_size(key, value)
                bytes_freed += item_size
                self.total_size -= item_size
            
            return len(expired_keys), bytes_freed
    
    def get_stats(self) -> CacheStats:
        """获取统计信息"""
        return CacheStats(
            total_entries=len(self.cache),
            access_count=self.stats.access_count,
            hit_count=self.stats.hit_count,
            eviction_count=self.stats.eviction_count
        )
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存使用信息"""
        return {
            'total_bytes': self.total_size,
            'max_bytes': self.max_memory_bytes,
            'usage_percent': (self.total_size / self.max_memory_bytes) * 100 if self.max_memory_bytes > 0 else 0,
            'avg_entry_size': self.total_size / len(self.cache) if len(self.cache) > 0 else 0
        }

class ShardedTTLCache:
    """分片TTL缓存 - 优化哈希分布"""
    
    def __init__(self, max_size: int = CacheConstants.MAX_SIZE, 
                 shard_count: int = CacheConstants.SHARD_COUNT,
                 default_ttl: int = CacheConstants.DEFAULT_TTL):
        self.default_ttl = default_ttl
        self.shard_count = shard_count
        self.shards: List[LRUCacheShard] = []
        
        # 初始化分片
        shard_size = max(1, max_size // shard_count)
        for i in range(shard_count):
            self.shards.append(LRUCacheShard(shard_size, shard_id=i))
        
        # 性能统计
        self._access_pattern: Dict[str, int] = {}
        self._pattern_limit = 1000
        
        # 定期清理任务
        self.cleanup_task: Optional[asyncio.Task] = None
        self.monitor_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
        self._start_monitor_task()
    
    def _get_shard(self, key: str) -> LRUCacheShard:
        """根据键获取对应的分片 - 使用更好的哈希分布"""
        if XXHASH_AVAILABLE:
            # 使用xxhash，更快且分布更好
            hash_value = xxhash.xxh64(key).intdigest()
        else:
            # 使用Python内置哈希，但加盐避免冲突
            hash_obj = hashlib.md5(key.encode())
            hash_value = int(hash_obj.hexdigest()[:8], 16)
        
        shard_index = hash_value % self.shard_count
        return self.shards[shard_index]
    
    def _record_access_pattern(self, key: str):
        """记录访问模式用于优化"""
        if len(self._access_pattern) >= self._pattern_limit:
            # 删除最旧的记录
            oldest_key = next(iter(self._access_pattern))
            del self._access_pattern[oldest_key]
        
        self._access_pattern[key] = self._access_pattern.get(key, 0) + 1
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        self._record_access_pattern(key)
        shard = self._get_shard(key)
        return await shard.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        self._record_access_pattern(key)
        shard = self._get_shard(key)
        ttl = ttl or self.default_ttl
        await shard.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """删除缓存键"""
        shard = self._get_shard(key)
        return await shard.delete(key)
    
    async def clear(self):
        """清空所有分片"""
        tasks = [shard.clear() for shard in self.shards]
        await asyncio.gather(*tasks)
        self._access_pattern.clear()
    
    async def cleanup_all_expired(self) -> Dict[str, Any]:
        """清理所有分片的过期条目"""
        tasks = [shard.cleanup_expired() for shard in self.shards]
        results = await asyncio.gather(*tasks)
        
        total_cleaned = 0
        total_bytes_freed = 0
        shard_details = []
        
        for i, (cleaned, bytes_freed) in enumerate(results):
            total_cleaned += cleaned
            total_bytes_freed += bytes_freed
            shard_details.append({
                'shard_id': i,
                'cleaned': cleaned,
                'bytes_freed': bytes_freed
            })
        
        return {
            'total_cleaned': total_cleaned,
            'total_bytes_freed': total_bytes_freed,
            'shard_details': shard_details
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取总体统计信息"""
        stats_tasks = [asyncio.create_task(self._get_shard_stats(i)) 
                      for i in range(self.shard_count)]
        shard_stats = await asyncio.gather(*stats_tasks)
        
        total_stats = CacheStats()
        total_memory = 0
        max_memory = 0
        
        for i, (stats, memory_info) in enumerate(shard_stats):
            total_stats.total_entries += stats.total_entries
            total_stats.access_count += stats.access_count
            total_stats.hit_count += stats.hit_count
            total_stats.eviction_count += stats.eviction_count
            total_memory += memory_info['total_bytes']
            max_memory += memory_info['max_bytes']
        
        # 分析访问模式
        hot_keys = sorted(self._access_pattern.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_entries": total_stats.total_entries,
            "access_count": total_stats.access_count,
            "hit_count": total_stats.hit_count,
            "eviction_count": total_stats.eviction_count,
            "hit_rate": total_stats.hit_rate,
            "shard_count": self.shard_count,
            "memory_usage": {
                "total_bytes": total_memory,
                "max_bytes": max_memory,
                "usage_percent": (total_memory / max_memory * 100) if max_memory > 0 else 0
            },
            "hot_keys": [{"key": k, "access_count": v} for k, v in hot_keys],
            "access_pattern_size": len(self._access_pattern)
        }
    
    async def _get_shard_stats(self, shard_index: int) -> Tuple[CacheStats, Dict[str, Any]]:
        """获取分片统计和内存信息"""
        shard = self.shards[shard_index]
        stats = shard.get_stats()
        memory_info = shard.get_memory_info()
        return stats, memory_info
    
    def _start_cleanup_task(self):
        """启动定期清理任务"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(TimeConstants.ONE_MINUTE * 5)  # 5分钟清理一次
                    result = await self.cleanup_all_expired()
                    if result['total_cleaned'] > 0:
                        print(f"缓存清理: 移除了 {result['total_cleaned']} 个过期条目，"
                              f"释放 {result['total_bytes_freed'] / 1024:.1f} KB")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"缓存清理任务错误: {e}")
                    await asyncio.sleep(TimeConstants.ONE_MINUTE)  # 出错后等待1分钟
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
    
    def _start_monitor_task(self):
        """启动监控任务"""
        async def monitor_loop():
            while True:
                try:
                    await asyncio.sleep(TimeConstants.ONE_MINUTE * 15)  # 15分钟报告一次
                    stats = await self.get_stats()
                    
                    if stats['hit_rate'] < 50:
                        print(f"缓存警告: 命中率较低 ({stats['hit_rate']:.1f}%)")
                    
                    if stats['memory_usage']['usage_percent'] > 80:
                        print(f"缓存警告: 内存使用率高 ({stats['memory_usage']['usage_percent']:.1f}%)")
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"缓存监控任务错误: {e}")
                    await asyncio.sleep(TimeConstants.ONE_MINUTE)
        
        self.monitor_task = asyncio.create_task(monitor_loop())
    
    async def close(self):
        """关闭缓存，清理资源"""
        # 取消清理任务
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 取消监控任务
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # 清理所有缓存
        await self.clear()
        
        print("缓存系统已安全关闭")

    async def get_shard_distribution(self) -> Dict[str, Any]:
        """获取分片分布情况"""
        distribution = {}
        for i, shard in enumerate(self.shards):
            stats = shard.get_stats()
            memory_info = shard.get_memory_info()
            distribution[f"shard_{i}"] = {
                "entries": stats.total_entries,
                "memory_bytes": memory_info['total_bytes'],
                "memory_percent": memory_info['usage_percent']
            }
        return distribution