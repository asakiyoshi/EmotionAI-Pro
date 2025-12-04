# tests/test_basic.py
"""
基础单元测试
"""
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from ..models import EnhancedEmotionalState, EmotionalMetrics, InteractionStats, TextDescriptions
from ..storage import UserStateRepository, AtomicJSONStorage
from ..cache import ShardedTTLCache, LRUCacheShard
from ..config import PluginConfig

class TestModels:
    """模型测试"""
    
    def test_emotional_metrics(self):
        """测试情感指标"""
        emotions = EmotionalMetrics(joy=10, trust=20)
        assert emotions.joy == 10
        assert emotions.trust == 20
        
        # 测试更新
        emotions.apply_update({'joy': 5, 'trust': -3})
        assert emotions.joy == 15
        assert emotions.trust == 17
        
        # 测试主导情感
        dominant = emotions.get_dominant()
        assert dominant in ["喜悦", "信任"]
    
    def test_enhanced_emotional_state(self):
        """测试增强情感状态"""
        state = EnhancedEmotionalState(user_key="test_user")
        assert state.user_key == "test_user"
        assert state.favor == 0
        assert state.intimacy == 0
        
        # 测试序列化和反序列化
        state_dict = state.to_dict()
        new_state = EnhancedEmotionalState.from_dict(state_dict)
        assert new_state.user_key == "test_user"
        
        # 测试有效性检查
        assert state.is_valid() == True

class TestStorage:
    """存储测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.mark.asyncio
    async def test_atomic_storage(self, temp_dir):
        """测试原子存储"""
        storage = AtomicJSONStorage(temp_dir / "test.json")
        
        # 测试保存和加载
        test_data = {"key": "value", "number": 123}
        await storage.save(test_data)
        
        loaded_data = await storage.load()
        assert loaded_data == test_data
    
    @pytest.mark.asyncio
    async def test_user_state_repository(self, temp_dir):
        """测试用户状态仓库"""
        repo = UserStateRepository(temp_dir)
        
        # 测试保存和加载用户状态
        state = EnhancedEmotionalState(user_key="test_user", favor=10, intimacy=20)
        await repo.save_user_state("test_user", state)
        
        loaded_state = await repo.get_user_state("test_user")
        assert loaded_state is not None
        assert loaded_state.favor == 10
        assert loaded_state.intimacy == 20

class TestCache:
    """缓存测试"""
    
    @pytest.mark.asyncio
    async def test_lru_cache_shard(self):
        """测试LRU缓存分片"""
        cache = LRUCacheShard(max_size=3)
        
        # 测试设置和获取
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        value1 = await cache.get("key1")
        assert value1 == "value1"
        
        value2 = await cache.get("key2")
        assert value2 == "value2"
        
        # 测试LRU淘汰
        await cache.set("key3", "value3")
        await cache.set("key4", "value4")  # 应该淘汰key1
        
        value1 = await cache.get("key1")
        assert value1 is None  # 已被淘汰
    
    @pytest.mark.asyncio
    async def test_sharded_cache(self):
        """测试分片缓存"""
        cache = ShardedTTLCache(max_size=10, shard_count=2)
        
        # 测试基本操作
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        value1 = await cache.get("key1")
        assert value1 == "value1"
        
        # 测试统计
        stats = await cache.get_stats()
        assert stats['total_entries'] >= 0

class TestConfig:
    """配置测试"""
    
    def test_plugin_config(self):
        """测试插件配置"""
        config = PluginConfig()
        assert config.session_based == False
        assert config.favour_min == -100
        assert config.favour_max == 100
        
        # 测试验证
        config_dict = config.dict()
        assert 'session_based' in config_dict
        assert 'favour_min' in config_dict
        
        # 测试更新
        new_config = PluginConfig(session_based=True, favour_min=-50)
        assert new_config.session_based == True
        assert new_config.favour_min == -50

if __name__ == "__main__":
    pytest.main([__file__, "-v"])