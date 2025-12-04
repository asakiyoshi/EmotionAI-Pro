# memory.py
import time
import asyncio
from typing import Dict, Any, Optional, List, Deque, Tuple
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
import heapq

from .cache import ShardedTTLCache
from .storage import UserStateRepository
from .constants import TimeConstants, UpdateThresholds

@dataclass
class InteractionRecord:
    """互动记录"""
    user_msg: str
    ai_response: str
    timestamp: float
    significance: int
    emotional_changes: Dict[str, int] = None
    embedding: Optional[List[float]] = None  # 可选的语义嵌入
    
    def __post_init__(self):
        if self.emotional_changes is None:
            self.emotional_changes = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'user_msg': self.user_msg,
            'ai_response': self.ai_response,
            'timestamp': self.timestamp,
            'significance': self.significance,
            'emotional_changes': self.emotional_changes,
            'embedding': self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InteractionRecord':
        """从字典创建"""
        return cls(
            user_msg=data.get('user_msg', ''),
            ai_response=data.get('ai_response', ''),
            timestamp=data.get('timestamp', 0),
            significance=data.get('significance', 0),
            emotional_changes=data.get('emotional_changes', {}),
            embedding=data.get('embedding')
        )

class EnhancedMemorySystem:
    """增强记忆系统 - 完全优化版本"""
    
    def __init__(self, repository: UserStateRepository, max_long_term_memory: int = 1000):
        self.repository = repository
        self.max_long_term_memory = max_long_term_memory
        
        # 短期记忆缓存 - 按用户分组
        self.short_term_memory = ShardedTTLCache(
            max_size=100,  # 100个用户
            default_ttl=TimeConstants.ONE_HOUR * 2  # 2小时
        )
        
        # 重要事件记忆 - 全局优先队列
        self.important_events: List[Tuple[int, float, Dict[str, Any]]] = []  # (-significance, timestamp, event)
        self.important_events_lock = asyncio.Lock()
        
        # 内存中的长期记忆 - 按用户组织
        self._long_term_memory: Dict[str, Deque[InteractionRecord]] = {}
        self._memory_loaded = False
        self._memory_lock = asyncio.Lock()
        
        # 记忆索引 - 用于快速检索
        self._memory_index: Dict[str, List[Tuple[float, str]]] = {}  # user_key -> [(timestamp, record_key)]
        
        # 清理任务
        self.cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
        
        print(f"增强记忆系统初始化完成，长期记忆限制: {max_long_term_memory}条")
    
    async def _ensure_memory_loaded(self):
        """确保记忆数据已加载"""
        async with self._memory_lock:
            if not self._memory_loaded:
                try:
                    memory_data = await self.repository.get_memory_data()
                    self._long_term_memory = self._deserialize_memory_data(memory_data)
                    
                    # 构建索引
                    self._build_memory_index()
                    
                    self._memory_loaded = True
                    print(f"加载了 {len(self._long_term_memory)} 个用户的长期记忆")
                    
                except Exception as e:
                    print(f"加载记忆数据失败: {e}")
                    self._long_term_memory = {}
                    self._memory_loaded = True
    
    def _deserialize_memory_data(self, memory_data: Dict[str, Any]) -> Dict[str, Deque[InteractionRecord]]:
        """反序列化记忆数据"""
        result = {}
        for user_key, records in memory_data.items():
            if isinstance(records, list):
                record_deque = deque(maxlen=50)  # 每个用户最多保留50条
                for record_data in records[-50:]:  # 只取最近50条
                    try:
                        record = InteractionRecord.from_dict(record_data)
                        record_deque.append(record)
                    except Exception as e:
                        print(f"反序列化记忆记录失败: {e}")
                        continue
                result[user_key] = record_deque
        return result
    
    def _serialize_memory_data(self) -> Dict[str, Any]:
        """序列化记忆数据"""
        result = {}
        for user_key, records in self._long_term_memory.items():
            result[user_key] = [record.to_dict() for record in records]
        return result
    
    def _build_memory_index(self):
        """构建记忆索引"""
        self._memory_index.clear()
        for user_key, records in self._long_term_memory.items():
            if user_key not in self._memory_index:
                self._memory_index[user_key] = []
            
            for record in records:
                # 使用组合键: user_key + timestamp
                record_key = f"{user_key}_{record.timestamp}"
                self._memory_index[user_key].append((record.timestamp, record_key))
            
            # 按时间戳排序
            self._memory_index[user_key].sort(key=lambda x: x[0], reverse=True)
    
    def _start_cleanup_task(self):
        """启动记忆清理任务"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(TimeConstants.ONE_HOUR)  # 每小时清理一次
                    await self._cleanup_memory()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"记忆清理任务错误: {e}")
                    await asyncio.sleep(TimeConstants.ONE_MINUTE * 5)
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_memory(self):
        """清理记忆"""
        try:
            # 清理短期记忆中的过期条目
            cleanup_result = await self.short_term_memory.cleanup_all_expired()
            if cleanup_result['total_cleaned'] > 0:
                print(f"清理短期记忆: {cleanup_result['total_cleaned']}条")
            
            # 检查长期记忆大小
            total_records = sum(len(records) for records in self._long_term_memory.values())
            if total_records > self.max_long_term_memory:
                await self._prune_long_term_memory()
            
        except Exception as e:
            print(f"记忆清理失败: {e}")
    
    async def _prune_long_term_memory(self):
        """修剪长期记忆"""
        try:
            # 收集所有记录进行排序
            all_records = []
            for user_key, records in self._long_term_memory.items():
                for record in records:
                    # 使用负的重要性分数进行最小堆排序
                    heapq.heappush(all_records, (-record.significance, record.timestamp, user_key, record))
            
            # 计算需要保留的记录数
            target_size = int(self.max_long_term_memory * 0.8)  # 保留80%
            
            # 重建长期记忆
            new_long_term_memory = {}
            records_kept = 0
            
            while all_records and records_kept < target_size:
                _, _, user_key, record = heapq.heappop(all_records)
                
                if user_key not in new_long_term_memory:
                    new_long_term_memory[user_key] = deque(maxlen=50)
                
                new_long_term_memory[user_key].append(record)
                records_kept += 1
            
            # 更新内存
            self._long_term_memory = new_long_term_memory
            
            # 重建索引
            self._build_memory_index()
            
            print(f"修剪长期记忆完成: {records_kept}条记录被保留")
            
        except Exception as e:
            print(f"修剪长期记忆失败: {e}")
    
    async def add_interaction(self, user_key: str, user_msg: str, 
                            ai_response: str, emotional_significance: int,
                            emotional_changes: Dict[str, int] = None):
        """添加互动到记忆系统"""
        try:
            await self._ensure_memory_loaded()
            
            # 创建新互动记录
            new_interaction = InteractionRecord(
                user_msg=user_msg[:500],  # 限制长度
                ai_response=ai_response[:500],
                timestamp=time.time(),
                significance=emotional_significance,
                emotional_changes=emotional_changes or {}
            )
            
            # 添加到短期记忆
            await self._add_to_short_term_memory(user_key, new_interaction)
            
            # 如果情感意义重大，存入长期记忆
            if emotional_significance >= UpdateThresholds.EMOTIONAL_SIGNIFICANCE:
                await self._add_to_long_term_memory(user_key, new_interaction)
                
                # 添加到重要事件
                async with self.important_events_lock:
                    heapq.heappush(self.important_events, 
                                 (-emotional_significance, new_interaction.timestamp, {
                                     'user_key': user_key,
                                     'user_msg': user_msg[:100],
                                     'significance': emotional_significance,
                                     'timestamp': new_interaction.timestamp
                                 }))
                    
                    # 保持重要事件数量
                    if len(self.important_events) > 50:
                        heapq.heappop(self.important_events)
                
                # 保存长期记忆
                await self._save_long_term_memory()
                
                print(f"添加到长期记忆 - 用户: {user_key}, 意义: {emotional_significance}/10")
            
        except Exception as e:
            print(f"添加互动到记忆系统失败: {e}")
            # 不重新抛出异常，避免影响主流程
    
    async def _add_to_short_term_memory(self, user_key: str, interaction: InteractionRecord):
        """添加到短期记忆"""
        try:
            # 获取现有近期互动
            recent_interactions = await self.short_term_memory.get(f"recent_{user_key}")
            if recent_interactions is None:
                recent_interactions = []
            
            # 添加到列表
            recent_interactions.append(interaction)
            
            # 只保留最近10条
            if len(recent_interactions) > 10:
                recent_interactions = recent_interactions[-10:]
            
            # 保存回缓存
            await self.short_term_memory.set(f"recent_{user_key}", recent_interactions)
            
        except Exception as e:
            print(f"添加到短期记忆失败: {e}")
    
    async def _add_to_long_term_memory(self, user_key: str, interaction: InteractionRecord):
        """添加到长期记忆"""
        async with self._memory_lock:
            if user_key not in self._long_term_memory:
                self._long_term_memory[user_key] = deque(maxlen=50)  # 每个用户最多50条
            
            # 添加到用户记忆
            self._long_term_memory[user_key].append(interaction)
            
            # 更新索引
            if user_key not in self._memory_index:
                self._memory_index[user_key] = []
            
            record_key = f"{user_key}_{interaction.timestamp}"
            self._memory_index[user_key].append((interaction.timestamp, record_key))
            
            # 保持索引排序
            self._memory_index[user_key].sort(key=lambda x: x[0], reverse=True)
    
    async def _save_long_term_memory(self):
        """保存长期记忆"""
        try:
            async with self._memory_lock:
                memory_data = self._serialize_memory_data()
                await self.repository.save_memory_data(memory_data)
        except Exception as e:
            print(f"保存长期记忆失败: {e}")
    
    def get_relationship_context(self, user_key: str) -> str:
        """获取关系上下文（长期记忆）- 修复同步方法"""
        try:
            long_term = self._long_term_memory.get(user_key, [])
            
            # 重要事件计数 - 现在使用同步方式
            important_count = 0
            important_events_copy = []
            
            # 复制重要事件列表以避免竞争条件
            try:
                # 注意：这里不能使用async with，所以我们需要其他方式
                # 创建一个临时副本
                if hasattr(self, 'important_events_lock'):
                    # 如果是异步环境，可以通过事件循环获取
                    import asyncio
                    try:
                        # 尝试同步获取锁，如果可能的话
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # 事件循环正在运行，我们需要其他方法
                            # 创建一个同步副本
                            important_events_copy = self.important_events.copy()
                        else:
                            # 事件循环没有运行，可以直接访问
                            important_events_copy = self.important_events.copy()
                    except:
                        # 出错时使用空列表
                        important_events_copy = []
                else:
                    important_events_copy = self.important_events.copy()
            except Exception:
                important_events_copy = []
            
            # 统计重要事件
            for _, _, event in important_events_copy:
                if event.get('user_key') == user_key:
                    important_count += 1
            
            context = "【长期关系发展轨迹】\n"
            if long_term:
                context += f"深度互动次数: {len(long_term)}\n"
                
                # 计算平均情感意义
                if long_term:
                    avg_significance = sum(item.significance for item in long_term) / len(long_term)
                    context += f"平均情感深度: {avg_significance:.1f}/10\n"
                
                # 最近的重要互动
                recent_significant = [item for item in long_term if item.significance >= 7]
                if recent_significant:
                    context += f"近期重要互动: {len(recent_significant)}次\n"
            
            if important_count > 0:
                context += f"重要时刻: {important_count}个\n"
            
            if not long_term and important_count == 0:
                context += "暂无长期互动记录\n"
            
            return context
            
        except Exception as e:
            print(f"获取关系上下文失败: {e}")
            return "【长期关系】记忆系统暂时不可用\n"
    
    async def get_relationship_context_async(self, user_key: str) -> str:
        """异步版本的关系上下文获取"""
        try:
            long_term = self._long_term_memory.get(user_key, [])
            
            # 重要事件计数 - 异步版本
            important_count = 0
            
            async with self.important_events_lock:
                for _, _, event in self.important_events:
                    if event.get('user_key') == user_key:
                        important_count += 1
            
            context = "【长期关系发展轨迹】\n"
            if long_term:
                context += f"深度互动次数: {len(long_term)}\n"
                
                if long_term:
                    avg_significance = sum(item.significance for item in long_term) / len(long_term)
                    context += f"平均情感深度: {avg_significance:.1f}/10\n"
                
                recent_significant = [item for item in long_term if item.significance >= 7]
                if recent_significant:
                    context += f"近期重要互动: {len(recent_significant)}次\n"
            
            if important_count > 0:
                context += f"重要时刻: {important_count}个\n"
            
            if not long_term and important_count == 0:
                context += "暂无长期互动记录\n"
            
            return context
            
        except Exception as e:
            print(f"获取关系上下文失败: {e}")
            return "【长期关系】记忆系统暂时不可用\n"
    
    async def get_recent_context(self, user_key: str) -> str:
        """获取近期对话上下文"""
        try:
            recent = await self.short_term_memory.get(f"recent_{user_key}")
            
            if recent is None:
                return "暂无近期对话记忆"
            
            if not isinstance(recent, list):
                print(f"近期对话数据格式错误，期望list，实际为{type(recent)}")
                return "近期对话记忆格式异常"
            
            if not recent:
                return "暂无近期对话"
            
            context = "【近期对话记忆】\n"
            for i, interaction in enumerate(recent[-5:], 1):  # 最多显示5条
                try:
                    user_msg = str(interaction.user_msg)
                    if len(user_msg) > 30:
                        user_msg = user_msg[:27] + "..."
                    
                    significance = interaction.significance
                    time_str = time.strftime("%H:%M", time.localtime(interaction.timestamp))
                    
                    context += f"{i}. [{time_str}] 用户: {user_msg}\n"
                    context += f"   情感意义: {significance}/10\n"
                except Exception as e:
                    print(f"处理单条互动记录时出错: {e}")
                    continue
            
            return context
            
        except Exception as e:
            print(f"获取近期上下文失败，用户{user_key}，错误: {e}")
            return "获取近期对话失败"
    
    async def get_user_memory_stats(self, user_key: str) -> Dict[str, Any]:
        """获取用户记忆统计"""
        await self._ensure_memory_loaded()
        
        try:
            long_term = self._long_term_memory.get(user_key, [])
            recent = await self.short_term_memory.get(f"recent_{user_key}") or []
            
            # 计算重要事件数量
            important_count = 0
            async with self.important_events_lock:
                for _, _, event in self.important_events:
                    if event['user_key'] == user_key:
                        important_count += 1
            
            # 计算统计信息
            total_significance = sum(item.significance for item in long_term)
            avg_significance = total_significance / len(long_term) if long_term else 0
            
            # 按情感分类
            emotion_categories = {
                'positive': 0,
                'negative': 0,
                'intimate': 0,
                'neutral': 0
            }
            
            for item in long_term:
                changes = item.emotional_changes
                if changes:
                    # 简单分类
                    positive_changes = sum(v for v in changes.values() if v > 0)
                    negative_changes = sum(abs(v) for v in changes.values() if v < 0)
                    
                    if positive_changes > negative_changes * 2:
                        emotion_categories['positive'] += 1
                    elif negative_changes > positive_changes * 2:
                        emotion_categories['negative'] += 1
                    elif 'intimacy' in changes and changes['intimacy'] > 0:
                        emotion_categories['intimate'] += 1
                    else:
                        emotion_categories['neutral'] += 1
            
            return {
                'long_term_count': len(long_term),
                'recent_count': len(recent),
                'important_count': important_count,
                'avg_significance': avg_significance,
                'total_significance': total_significance,
                'emotion_distribution': emotion_categories,
                'last_interaction': long_term[-1].timestamp if long_term else 0
            }
            
        except Exception as e:
            print(f"获取用户记忆统计失败: {e}")
            return {
                'long_term_count': 0,
                'recent_count': 0,
                'important_count': 0,
                'avg_significance': 0,
                'total_significance': 0,
                'emotion_distribution': {},
                'last_interaction': 0
            }
    
    async def search_memories(self, user_key: str, query: str = "", 
                            start_time: float = 0, end_time: float = 0) -> List[InteractionRecord]:
        """搜索记忆"""
        await self._ensure_memory_loaded()
        
        try:
            memories = self._long_term_memory.get(user_key, [])
            results = []
            
            for memory in memories:
                # 时间过滤
                if start_time > 0 and memory.timestamp < start_time:
                    continue
                if end_time > 0 and memory.timestamp > end_time:
                    continue
                
                # 关键词搜索
                if query:
                    query_lower = query.lower()
                    if (query_lower in memory.user_msg.lower() or 
                        query_lower in memory.ai_response.lower()):
                        results.append(memory)
                else:
                    results.append(memory)
            
            # 按时间倒序排序
            results.sort(key=lambda x: x.timestamp, reverse=True)
            
            return results
            
        except Exception as e:
            print(f"搜索记忆失败: {e}")
            return []
    
    async def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆系统摘要"""
        await self._ensure_memory_loaded()
        
        try:
            # 短期记忆统计
            cache_stats = await self.short_term_memory.get_stats()
            
            # 长期记忆统计
            total_users = len(self._long_term_memory)
            total_records = sum(len(records) for records in self._long_term_memory.values())
            
            # 重要事件统计
            async with self.important_events_lock:
                important_count = len(self.important_events)
                avg_importance = sum(-score for score, _, _ in self.important_events) / important_count if important_count > 0 else 0
            
            return {
                'short_term_memory': {
                    'total_entries': cache_stats.get('total_entries', 0),
                    'memory_usage': cache_stats.get('memory_usage', {})
                },
                'long_term_memory': {
                    'total_users': total_users,
                    'total_records': total_records,
                    'avg_records_per_user': total_records / total_users if total_users > 0 else 0
                },
                'important_events': {
                    'count': important_count,
                    'avg_importance': avg_importance
                },
                'system': {
                    'max_long_term_memory': self.max_long_term_memory,
                    'memory_loaded': self._memory_loaded
                }
            }
            
        except Exception as e:
            print(f"获取记忆系统摘要失败: {e}")
            return {}
    
    async def close(self):
        """关闭记忆系统"""
        print("正在关闭记忆系统...")
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 保存长期记忆
        await self._save_long_term_memory()
        
        print("记忆系统已关闭")