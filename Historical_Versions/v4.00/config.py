# config.py
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class PrivacyLevel(int, Enum):
    FULL_SECRET = 0
    BASIC = 1  
    DETAILED = 2

class AttitudeType(str, Enum):
    HOSTILE = "敌对"
    COLD = "冷淡" 
    NEUTRAL = "中立"
    FRIENDLY = "友好"
    INTIMATE = "热情"

class RelationshipStage(str, Enum):
    INITIAL = "初识期"
    DEEPENING = "深化期"
    COMMITMENT = "承诺期" 
    SYMBIOSIS = "共生期"

class PluginConfig(BaseModel):
    """插件配置模型"""
    session_based: bool = Field(default=False, description="是否启用会话独立的情感系统")
    favour_min: int = Field(default=-100, ge=-1000, le=0, description="好感度最小值")
    favour_max: int = Field(default=100, ge=0, le=1000, description="好感度最大值")
    intimacy_min: int = Field(default=0, ge=0, le=100, description="亲密度最小值")
    intimacy_max: int = Field(default=100, ge=0, le=1000, description="亲密度最大值")
    change_min: int = Field(default=-10, ge=-100, le=0, description="好感度单次变化最小值")
    change_max: int = Field(default=5, ge=0, le=100, description="好感度单次变化最大值")
    admin_qq_list: List[str] = Field(default_factory=list, description="管理员QQ号列表")
    plugin_priority: int = Field(default=100000, ge=1, le=1000000, description="插件处理优先级")
    enable_attitude_system: bool = Field(default=True, description="启用态度关系系统")
    enable_ai_text_generation: bool = Field(default=True, description="启用AI自主生成文本描述")
    global_privacy_level: PrivacyLevel = Field(default=PrivacyLevel.BASIC, description="全局隐私级别")
    enable_smart_update: bool = Field(default=True, description="启用智能更新机制")
    force_update_interval: int = Field(default=5, ge=1, le=100, description="强制更新间隔（对话次数）")
    emotional_significance_threshold: int = Field(default=5, ge=1, le=10, description="情感意义阈值")
    enable_secondary_llm: bool = Field(default=True, description="启用辅助LLM进行情感分析")
    secondary_llm_provider: Optional[str] = Field(default=None, description="辅助LLM提供商")
    secondary_llm_model: Optional[str] = Field(default=None, description="辅助LLM模型名称")
    
    # 性能配置
    cache_ttl: int = Field(default=300, description="缓存默认TTL(秒)")
    cache_max_size: int = Field(default=1000, description="缓存最大条目数")
    auto_save_interval: int = Field(default=60, description="自动保存间隔(秒)")
    max_dirty_keys: int = Field(default=1000, description="最大脏键数量")
    backup_retention_days: int = Field(default=7, description="备份保留天数")

    class Config:
        use_enum_values = True