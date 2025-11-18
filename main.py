import json
import re
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import asyncio

from dataclasses import dataclass, asdict
from enum import Enum

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api import AstrBotConfig, logger


# ==================== 枚举定义 ====================

class PrivacyLevel(Enum):
    """隐私级别枚举"""
    FULL_SECRET = 0    # 完全保密
    BASIC = 1          # 基础显示  
    DETAILED = 2       # 详细显示


class AttitudeType(Enum):
    """态度类型枚举"""
    HOSTILE = "敌对"      # -100 ~ -51
    COLD = "冷淡"         # -50 ~ -11  
    NEUTRAL = "中立"      # -10 ~ 39
    FRIENDLY = "友好"     # 40 ~ 74
    INTIMATE = "热情"     # 75 ~ 100


class RelationshipStage(Enum):
    """关系发展阶段枚举"""
    INITIAL = "初识期"           # 好感驱动，建立吸引
    DEEPENING = "深化期"         # 互动平衡，共同成长  
    COMMITMENT = "承诺期"        # 亲密主导，根基稳固
    SYMBIOSIS = "共生期"         # 完全融合，不分彼此


# ==================== 数据结构定义 ====================

@dataclass
class EnhancedEmotionalState:
    """增强的情感状态 - 融合两个系统的精华"""
    
    # EmotionAI 的8维情感
    joy: int = 0
    trust: int = 0
    fear: int = 0
    surprise: int = 0
    sadness: int = 0
    disgust: int = 0
    anger: int = 0
    anticipation: int = 0
    
    # EmotionAI 的复合状态
    favor: int = 0
    intimacy: int = 0
    
    # FavourPro 的核心系统（强化版）
    attitude: str = "中立"
    relationship: str = "陌生人"
    attitude_intensity: int = 0  # -10 到 +10
    relationship_depth: int = 0  # 0-100
    
    # 行为统计
    interaction_count: int = 0
    last_interaction: float = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    
    # 用户设置
    show_status: bool = False
    privacy_level: int = None   # 用户级隐私
    
    # 关系阶段追踪
    relationship_stage: str = "初识期"
    stage_composite_score: float = 0.0
    stage_progress: float = 0.0  # 到下一阶段的进度百分比
    
    # 新增：文本更新追踪
    last_attitude_update: float = 0
    last_relationship_update: float = 0
    attitude_update_count: int = 0
    relationship_update_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedEmotionalState':
        """从字典创建实例，处理字段不匹配问题"""
        valid_fields = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # 记录被过滤掉的字段（用于调试）
        removed_fields = set(data.keys()) - set(valid_fields)
        if removed_fields:
            logger.debug(f"过滤掉不兼容字段: {removed_fields}")
        
        return cls(**filtered_data)
    
    def update_stage_info(self, stage_info: Dict[str, Any]):
        """更新阶段信息"""
        self.relationship_stage = stage_info["stage_name"]
        self.stage_composite_score = stage_info["composite_score"] 
        self.stage_progress = stage_info["progress_to_next"]
    
    def should_update_text_descriptions(self, force_update: bool = False) -> bool:
        """判断是否需要更新文本描述"""
        current_time = time.time()
        
        # 如果强制更新，总是返回True
        if force_update:
            return True
            
        # 基于时间间隔（最多5分钟不更新就强制更新）
        time_since_attitude = current_time - self.last_attitude_update
        time_since_relationship = current_time - self.last_relationship_update
        
        if time_since_attitude > 300 or time_since_relationship > 300:  # 5分钟
            return True
            
        # 基于互动计数（每3次互动至少更新一次）
        if self.interaction_count % 3 == 0:
            return True
            
        return False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedEmotionalState':
        """从字典创建实例，处理字段不匹配问题"""
        valid_fields = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # 记录被过滤掉的字段（用于调试）
        removed_fields = set(data.keys()) - set(valid_fields)
        if removed_fields:
            logger.debug(f"过滤掉不兼容字段: {removed_fields}")
        
        return cls(**filtered_data)
    
    def update_stage_info(self, stage_info: Dict[str, Any]):
        """更新阶段信息"""
        self.relationship_stage = stage_info["stage_name"]
        self.stage_composite_score = stage_info["composite_score"] 
        self.stage_progress = stage_info["progress_to_next"]


@dataclass  
class RankingEntry:
    """排行榜条目"""
    rank: int
    user_key: str
    average_score: float
    favor: int
    intimacy: int
    attitude: str
    relationship: str
    display_name: str


# ==================== 核心管理器类 ====================

class AttitudeRelationshipManager:
    """态度关系管理器 - 基于好感度和亲密度的复合评估"""
    
    # 行为风格映射
    BEHAVIOR_STYLES = {
        "hostile": "极其简短、尖锐，可能拒绝回应",
        "cold": "冷淡、简洁、不耐烦", 
        "neutral": "客观、保持距离、标准化回应",
        "friendly": "积极、乐于协助、带有正面情绪",
        "intimate": "热情、主动、富有情感，可使用亲昵称呼"
    }
    
    @classmethod
    def get_behavior_guidance(cls, favor: int, intimacy: int) -> str:
        """根据好感度和亲密度获取行为指导"""
        # 复合评分 = 好感度 * 权重 + 亲密度 * 权重
        composite_score = favor * 0.6 + intimacy * 0.4
        
        if composite_score >= 75:
            return cls.BEHAVIOR_STYLES["intimate"]
        elif composite_score >= 40:
            return cls.BEHAVIOR_STYLES["friendly"]
        elif composite_score >= -10:
            return cls.BEHAVIOR_STYLES["neutral"]
        elif composite_score >= -50:
            return cls.BEHAVIOR_STYLES["cold"]
        else:
            return cls.BEHAVIOR_STYLES["hostile"]
    
    @classmethod
    def should_ai_update_descriptions(cls, current_state: EnhancedEmotionalState, 
                                    new_favor: int, new_intimacy: int) -> bool:
        """判断是否需要 AI 更新文本描述 - 大幅提高更新频率"""
        # 基于复合评分变化判断
        old_composite = current_state.favor * 0.6 + current_state.intimacy * 0.4
        new_composite = new_favor * 0.6 + new_intimacy * 0.4
        composite_change = abs(new_composite - old_composite)
        
        # 大幅提高更新频率：复合评分变化阈值从5降低到1，互动计数模数从3降低到1
        if (composite_change >= 1 or  # 从5降低到1
            current_state.interaction_count % 1 == 0 or  # 从3降低到1（每次互动都检查）
            time.time() - current_state.last_interaction > 1 * 24 * 3600):  # 从3天降低到1天
            return True
        return False


class DynamicWeightManager:
    """动态权重管理器 - 基于关系发展阶段（带过渡保护）"""
    
    # 关系阶段配置 - 增加过渡期配置
    STAGE_CONFIGS = {
        RelationshipStage.INITIAL: {
            "name": "初识期",
            "description": "好感驱动，建立吸引",
            "favor_weight": 0.7,
            "intimacy_weight": 0.3,
            "favor_range": (0, 40),
            "intimacy_range": (0, 30),
            "composite_threshold": 25,
            "transition_buffer": 3,  # 过渡缓冲区间
            "intimacy_boost_factor": 4.0  # 亲密度提升倍率
        },
        RelationshipStage.DEEPENING: {
            "name": "深化期", 
            "description": "互动平衡，共同成长",
            "favor_weight": 0.5,
            "intimacy_weight": 0.5,
            "favor_range": (40, 70),
            "intimacy_range": (30, 60),
            "composite_threshold": 55,
            "transition_buffer": 5,
            "intimacy_boost_factor": 3.6
        },
        RelationshipStage.COMMITMENT: {
            "name": "承诺期",
            "description": "亲密主导，根基稳固", 
            "favor_weight": 0.3,
            "intimacy_weight": 0.7,
            "favor_range": (70, 90),
            "intimacy_range": (60, 85),
            "composite_threshold": 80,
            "transition_buffer": 7,
            "intimacy_boost_factor": 3.0
        },
        RelationshipStage.SYMBIOSIS: {
            "name": "共生期",
            "description": "完全融合，不分彼此",
            "favor_weight": 0.5,
            "intimacy_weight": 0.5,
            "favor_range": (90, 100),
            "intimacy_range": (85, 100),
            "composite_threshold": 95,
            "transition_buffer": 10,
            "intimacy_boost_factor": 1.0  # 共生期不再需要特别提升
        }
    }
    
    @classmethod
    def calculate_stage(cls, state: EnhancedEmotionalState) -> Tuple[RelationshipStage, Dict[str, Any]]:
        """计算当前关系阶段和过渡状态"""
        current_composite = cls._calculate_raw_composite(state)
        previous_stage = getattr(state, '_previous_stage', RelationshipStage.INITIAL)
        previous_composite = getattr(state, '_previous_composite', 0.0)
        
        # 判断当前阶段
        target_stage = cls._get_stage_by_score(current_composite, state)
        
        # 检查是否处于阶段过渡期
        transition_info = cls._check_transition_status(
            state, previous_stage, target_stage, previous_composite, current_composite
        )
        
        return target_stage, transition_info
    
    @classmethod
    def _calculate_raw_composite(cls, state: EnhancedEmotionalState) -> float:
        """计算原始复合评分（不使用过渡保护）"""
        current_stage = cls._get_stage_by_score(
            state.favor * 0.6 + state.intimacy * 0.4,
            state
        )
        stage_config = cls.STAGE_CONFIGS[current_stage]
        return state.favor * stage_config["favor_weight"] + state.intimacy * stage_config["intimacy_weight"]
    
    @classmethod
    def _get_stage_by_score(cls, composite_score: float, state: EnhancedEmotionalState) -> RelationshipStage:
        """滞后版阶段判定：上升阈值 > 下降阈值，防止抖动"""
        # 读取上一次阶段
        prev_stage = getattr(state, '_previous_stage', RelationshipStage.INITIAL)

        # 计算当前“裸”阶段
        if composite_score >= cls.STAGE_CONFIGS[RelationshipStage.SYMBIOSIS]["composite_threshold"]:
            raw_target = RelationshipStage.SYMBIOSIS
        elif composite_score >= cls.STAGE_CONFIGS[RelationshipStage.COMMITMENT]["composite_threshold"]:
            raw_target = RelationshipStage.COMMITMENT
        elif composite_score >= cls.STAGE_CONFIGS[RelationshipStage.DEEPENING]["composite_threshold"]:
            raw_target = RelationshipStage.DEEPENING
        else:
            raw_target = RelationshipStage.INITIAL

        # ---------- 滞后逻辑 ----------
        UP_THRESHOLD   = cls.STAGE_CONFIGS[raw_target]["composite_threshold"]
        DOWN_THRESHOLD = UP_THRESHOLD - 5          # 5 点滞后带
        # 如果比上一阶段高，用上升阈值；否则用下降阈值
        use_threshold  = UP_THRESHOLD if raw_target.value > prev_stage.value else DOWN_THRESHOLD

        if composite_score < use_threshold:        # 未达标，留在原阶段
            return prev_stage

        # 达标，允许进入新阶段
        return raw_target
    
    @classmethod
    def _check_transition_status(cls, state: EnhancedEmotionalState, previous_stage: RelationshipStage, 
                               target_stage: RelationshipStage, previous_composite: float, 
                               current_composite: float) -> Dict[str, Any]:
        """检查过渡状态并应用保护机制"""
        transition_info = {
            "is_transitioning": False,
            "from_stage": previous_stage,
            "to_stage": target_stage,
            "protected_composite": current_composite,
            "intimacy_boost_active": False,
            "transition_progress": 0.0,
            "needed_intimacy_boost": 0
        }
        
        # 如果阶段发生变化，进入过渡期
        if previous_stage != target_stage:
            transition_info["is_transitioning"] = True
            
            # 应用复合评分保护：不低于前一阶段的最高评分
            stage_config = cls.STAGE_CONFIGS[previous_stage]
            protected_score = max(current_composite, previous_composite)
            transition_info["protected_composite"] = protected_score
            
            # 计算需要的亲密度提升
            target_config = cls.STAGE_CONFIGS[target_stage]
            needed_intimacy = cls._calculate_needed_intimacy(state, target_config, protected_score)
            transition_info["needed_intimacy_boost"] = needed_intimacy
            transition_info["intimacy_boost_active"] = needed_intimacy > 0
            
            # 计算过渡进度
            transition_info["transition_progress"] = cls._calculate_transition_progress(
                state, target_config, needed_intimacy
            )
            
            logger.info(f"阶段过渡: {previous_stage.value} -> {target_stage.value}, "
                       f"保护评分: {protected_score:.1f}, 需要亲密度: {needed_intimacy}")
        
        return transition_info
    
    @classmethod
    def _calculate_needed_intimacy(cls, state: EnhancedEmotionalState, target_config: Dict[str, Any], 
                                 protected_score: float) -> int:
        """计算达到目标阶段所需的最小亲密度"""
        # 基于公式: protected_score = favor * fav_weight + intimacy * int_weight
        # 推导: intimacy = (protected_score - favor * fav_weight) / int_weight
        fav_weight = target_config["favor_weight"]
        int_weight = target_config["intimacy_weight"]
        
        if int_weight == 0:
            return 0
            
        needed_intimacy = (protected_score - state.favor * fav_weight) / int_weight
        needed_intimacy = max(0, needed_intimacy)  # 不能为负数
        needed_intimacy = min(100, needed_intimacy)  # 不能超过最大值
        
        # 取整并确保比当前亲密度高
        needed_intimacy = int(needed_intimacy)
        current_intimacy = state.intimacy
        
        return max(0, needed_intimacy - current_intimacy)
    
    @classmethod
    def _calculate_transition_progress(cls, state: EnhancedEmotionalState, target_config: Dict[str, Any],
                                    needed_intimacy: int) -> float:
        """计算过渡进度"""
        if needed_intimacy <= 0:
            return 100.0
            
        # 计算当前亲密度与目标亲密度的比例
        target_intimacy = state.intimacy + needed_intimacy
        current_progress = (state.intimacy / target_intimacy) * 100 if target_intimacy > 0 else 0
        return min(100.0, current_progress)
    
    @classmethod
    def get_stage_weights(cls, state: EnhancedEmotionalState) -> Tuple[float, float]:
        """获取当前阶段的权重（考虑过渡期）"""
        target_stage, transition_info = cls.calculate_stage(state)
        stage_config = cls.STAGE_CONFIGS[target_stage]
        
        # 如果在过渡期且需要亲密度提升，调整权重以促进亲密度增长
        if transition_info["intimacy_boost_active"]:
            boost_factor = stage_config["intimacy_boost_factor"]
            # 临时增加亲密度权重
            base_favor = stage_config["favor_weight"]
            base_intimacy = stage_config["intimacy_weight"]
            
            # 调整权重，但保持总和为1
            total = base_favor + base_intimacy * boost_factor
            adjusted_favor = base_favor / total
            adjusted_intimacy = (base_intimacy * boost_factor) / total
            
            logger.debug(f"过渡期权重调整: 亲密度权重 x{boost_factor}, "
                        f"调整后: 好感{adjusted_favor:.2f}, 亲密{adjusted_intimacy:.2f}")
            
            return adjusted_favor, adjusted_intimacy
        
        return stage_config["favor_weight"], stage_config["intimacy_weight"]
    
    @classmethod
    def calculate_composite_score(cls, state: EnhancedEmotionalState) -> float:
        """计算当前阶段的复合评分（应用过渡保护）"""
        target_stage, transition_info = cls.calculate_stage(state)
        return transition_info["protected_composite"]
    
    @classmethod
    def get_stage_info(cls, state: EnhancedEmotionalState) -> Dict[str, Any]:
        """获取完整的阶段信息（包含过渡状态）"""
        # 如果好感度为负，使用特殊处理
        if state.favor < 0:
            return cls._get_negative_favor_stage_info(state)
        
        target_stage, transition_info = cls.calculate_stage(state)
        stage_config = cls.STAGE_CONFIGS[target_stage]
        
        favor_weight, intimacy_weight = cls.get_stage_weights(state)
        composite_score = cls.calculate_composite_score(state)
        
        # 计算阶段进度，确保最小值不为负数
        progress = (composite_score / stage_config["composite_threshold"]) * 100
        progress_to_next = max(0, min(100, progress))  # 确保进度在0-100之间
        
        info = {
            "stage": target_stage,
            "stage_name": stage_config["name"],
            "description": stage_config["description"],
            "favor_weight": favor_weight,
            "intimacy_weight": intimacy_weight,
            "composite_score": composite_score,
            "next_stage_threshold": stage_config["composite_threshold"],
            "progress_to_next": progress_to_next,  # 使用修正后的进度
            "is_transitioning": transition_info["is_transitioning"],
            "transition_progress": transition_info["transition_progress"],
            "intimacy_boost_active": transition_info["intimacy_boost_active"],
            "needed_intimacy_boost": transition_info["needed_intimacy_boost"]
        }
        
        # 保存当前状态用于下一次计算
        state._previous_stage = target_stage
        state._previous_composite = composite_score
        
        return info
    
    @classmethod
    def _get_negative_favor_stage_info(cls, state: EnhancedEmotionalState) -> Dict[str, Any]:
        """获取负好感时的阶段信息"""
        # 负好感时，好感度占100%权重，亲密度不参与计算
        composite_score = state.favor
        
        # 根据负好感程度确定阶段
        if state.favor >= -30:
            stage_name = "冷淡期"
            description = "关系冷淡，需要修复"
            progress = max(0, (state.favor + 30) / 30 * 100)  # 从-30到0的进度
        elif state.favor >= -70:
            stage_name = "反感期"
            description = "存在反感情绪"
            progress = max(0, (state.favor + 70) / 40 * 100)  # 从-70到-30的进度
        else:
            stage_name = "敌对期"
            description = "关系敌对"
            progress = 0  # 极度负好感时进度为0
        
        return {
            "stage": None,  # 负好感阶段不属于正常阶段
            "stage_name": stage_name,
            "description": description,
            "favor_weight": 1.0,  # 负好感时好感度占100%权重
            "intimacy_weight": 0.0,  # 亲密度不参与计算
            "composite_score": composite_score,
            "next_stage_threshold": 0,  # 下一个阶段是中立状态
            "progress_to_next": progress,
            "is_transitioning": False,
            "transition_progress": 0.0,
            "intimacy_boost_active": False,
            "needed_intimacy_boost": 0
        }
    
    @classmethod
    def get_stage_weights(cls, state: EnhancedEmotionalState) -> Tuple[float, float]:
        """获取当前阶段的权重（考虑过渡期）"""
        # 如果好感度为负，使用特殊权重
        if state.favor < 0:
            return 1.0, 0.0  # 负好感时好感度占100%权重
        
        target_stage, transition_info = cls.calculate_stage(state)
        stage_config = cls.STAGE_CONFIGS[target_stage]
        
        # 如果在过渡期且需要亲密度提升，调整权重以促进亲密度增长
        if transition_info["intimacy_boost_active"]:
            boost_factor = stage_config["intimacy_boost_factor"]
            # 临时增加亲密度权重
            base_favor = stage_config["favor_weight"]
            base_intimacy = stage_config["intimacy_weight"]
            
            # 调整权重，但保持总和为1
            total = base_favor + base_intimacy * boost_factor
            adjusted_favor = base_favor / total
            adjusted_intimacy = (base_intimacy * boost_factor) / total
            
            logger.debug(f"过渡期权重调整: 亲密度权重 x{boost_factor}, "
                        f"调整后: 好感{adjusted_favor:.2f}, 亲密{adjusted_intimacy:.2f}")
            
            return adjusted_favor, adjusted_intimacy
        
        return stage_config["favor_weight"], stage_config["intimacy_weight"]
    
    @classmethod
    def calculate_composite_score(cls, state: EnhancedEmotionalState) -> float:
        """计算当前阶段的复合评分（应用过渡保护）"""
        # 如果好感度为负，只使用好感度
        if state.favor < 0:
            return state.favor
        
        target_stage, transition_info = cls.calculate_stage(state)
        return transition_info["protected_composite"]    
    
    @classmethod
    def apply_transition_benefits(cls, state: EnhancedEmotionalState, updates: Dict[str, Any]) -> Dict[str, Any]:
        """应用过渡期增益效果"""
        target_stage, transition_info = cls.calculate_stage(state)
        
        if transition_info["intimacy_boost_active"]:
            stage_config = cls.STAGE_CONFIGS[target_stage]
            boost_factor = stage_config["intimacy_boost_factor"]
            
            # 增强亲密度的增长效果
            if 'intimacy' in updates:
                original_boost = updates['intimacy']
                boosted_boost = int(original_boost * boost_factor)
                updates['intimacy'] = boosted_boost
                
                logger.info(f"过渡期亲密度增益: {original_boost} -> {boosted_boost} (x{boost_factor})")
            
            # 在过渡期，正面互动更容易提升亲密度
            if ('joy' in updates or 'trust' in updates or 'anticipation' in updates) and 'intimacy' not in updates:
                # 自动添加小额亲密度提升
                auto_intimacy = max(1, int(2 * boost_factor))
                updates['intimacy'] = updates.get('intimacy', 0) + auto_intimacy
                logger.debug(f"过渡期自动亲密度增益: +{auto_intimacy}")
        
        return updates
    
    @classmethod
    def get_stage_progression_advice(cls, state: EnhancedEmotionalState) -> str:
        """获取阶段进阶建议（包含过渡期建议）"""
        stage_info = cls.get_stage_info(state)
    
        # 负好感阶段的建议
        if state.favor < 0:
            if state.favor >= -30:
                return "冷淡期：需要真诚道歉和积极行动来修复关系，避免进一步恶化。"
            elif state.favor >= -70:
                return "反感期：需要时间和耐心来缓解负面情绪，避免直接冲突。"
            else:
                return "敌对期：关系极度紧张，需要保持距离或寻求第三方调解。"
    
        if stage_info["is_transitioning"]:
            if stage_info["intimacy_boost_active"]:
                return (f"【阶段过渡中】{stage_info['stage_name']}\n"
                        f"   当前需要提升亲密度 {stage_info['needed_intimacy_boost']} 点来适应新阶段\n"
                       f"   过渡进度: {stage_info['transition_progress']:.1f}%\n"
                       f"   建议: 多进行深度交流，分享个人经历和情感")
            else:
                return (f"【阶段过渡完成】{stage_info['stage_name']}\n"
                       f"   已成功进入新阶段，关系正在稳定发展")
    
        # 原有建议
        advice_map = {
            RelationshipStage.INITIAL: 
                "初识期：多展示个人魅力，建立良好第一印象。通过有趣的话题和积极的互动提升好感度。",
            RelationshipStage.DEEPENING:
                "深化期：分享更多个人经历和情感，建立信任基础。共同经历和深度交流是关键。", 
            RelationshipStage.COMMITMENT:
                "承诺期：巩固信任和默契，在困难时刻相互支持。关系的深度比广度更重要。",
            RelationshipStage.SYMBIOSIS:
                "共生期：维持情感的深度连接，共同成长和创造美好回忆。"
        }
    
        return advice_map.get(stage_info["stage"], "继续培养这段关系吧！")


class UserStateManager:
    """用户状态管理器"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.user_data = self._load_and_migrate_data("user_emotion_data.json")
        self.dirty_keys = set()
        self.last_save_time = time.time()
        self.save_interval = 60
        self.lock = asyncio.Lock()
    
    def _load_and_migrate_data(self, filename: str) -> Dict[str, Any]:
        """加载并迁移数据文件"""
        path = self.data_path / filename
        if not path.exists():
            return {}
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            # 数据迁移：移除不兼容的字段
            migrated_data = {}
            migration_count = 0
            
            for user_key, user_data in raw_data.items():
                if isinstance(user_data, dict):
                    # 移除 EnhancedEmotionalState 中不存在的字段
                    valid_fields = EnhancedEmotionalState.__annotations__.keys()
                    cleaned_data = {k: v for k, v in user_data.items() 
                                  if k in valid_fields}
                    
                    # 检查是否有字段被移除
                    if len(cleaned_data) != len(user_data):
                        migration_count += 1
                    
                    migrated_data[user_key] = cleaned_data
                else:
                    migrated_data[user_key] = user_data
            
            if migration_count > 0:
                logger.info(f"数据迁移完成，修复了 {migration_count} 个用户的数据格式")
            
            return migrated_data
            
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"数据文件加载失败: {e}")
            return {}
    
    async def get_user_state(self, user_key: str) -> EnhancedEmotionalState:
        """获取用户情感状态（带错误处理）"""
        async with self.lock:
            if user_key in self.user_data:
                try:
                    return EnhancedEmotionalState.from_dict(self.user_data[user_key])
                except TypeError as e:
                    logger.warning(f"用户 {user_key} 数据格式错误，重置为默认状态: {e}")
                    # 数据格式错误，返回默认状态并修复数据
                    default_state = EnhancedEmotionalState()
                    self.user_data[user_key] = default_state.to_dict()
                    self.dirty_keys.add(user_key)
                    return default_state
            return EnhancedEmotionalState()
    
    async def update_user_state(self, user_key: str, state: EnhancedEmotionalState):
        """更新用户状态"""
        async with self.lock:
            self.user_data[user_key] = state.to_dict()
            self.dirty_keys.add(user_key)
        await self._check_auto_save()
    
    async def _check_auto_save(self):
        """检查是否需要自动保存"""
        current_time = time.time()
        if (current_time - self.last_save_time >= self.save_interval and 
            self.dirty_keys):
            await self.force_save()
    
    async def force_save(self):
        """强制保存所有脏数据"""
        async with self.lock:
            if self.dirty_keys:
                self._save_data("user_emotion_data.json", self.user_data)
                self.dirty_keys.clear()
                self.last_save_time = time.time()
    
    def _save_data(self, filename: str, data: Dict[str, Any]):
        """保存数据到文件"""
        path = self.data_path / filename
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存数据失败: {e}")

    # ---------- 昵称映射工具（与 _save_data 平级） ----------
    def _load_nickname_map(self):
        path = self.data_path / "nickname_map.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _save_nickname_map(self, mapping):
        path = self.data_path / "nickname_map.json"
        path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    
    async def clear_all_data(self):
        """清空所有用户数据"""
        async with self.lock:
            self.user_data.clear()
            self.dirty_keys.clear()
            await self.force_save()


class TTLCache:
    """带过期时间的缓存"""
    
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.lock = asyncio.Lock()
        self.access_count = 0
        self.hit_count = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        async with self.lock:
            self.access_count += 1
            if key in self.cache:
                value, expires_at = self.cache[key]
                if time.time() < expires_at:
                    self.hit_count += 1
                    return value
                else:
                    del self.cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        async with self.lock:
            # 清理过期缓存
            await self._cleanup_expired()
            
            # 如果超过最大大小，删除最旧的
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl
            self.cache[key] = (value, expires_at)
    
    async def _cleanup_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expires_at) in self.cache.items()
            if current_time >= expires_at
        ]
        for key in expired_keys:
            del self.cache[key]
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        async with self.lock:
            hit_rate = (self.hit_count / self.access_count * 100) if self.access_count > 0 else 0
            return {
                "total_entries": len(self.cache),
                "access_count": self.access_count,
                "hit_count": self.hit_count,
                "hit_rate": round(hit_rate, 2)
            }
    
    async def clear(self):
        """清空缓存"""
        async with self.lock:
            self.cache.clear()


class RankingManager:
    """排行榜管理器"""
    
    def __init__(self, user_state_manager):
        self.user_state_manager = user_state_manager
        self.cache = TTLCache(default_ttl=60, max_size=10)
    
    async def get_enhanced_ranking(self, limit: int = 10, reverse: bool = True) -> List[RankingEntry]:
        """获取增强的排行榜"""
        cache_key = f"ranking_{limit}_{reverse}"
        
        # 尝试从缓存获取
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # 计算排行榜
        averages = []
        
        async with self.user_state_manager.lock:
            for user_key, data in self.user_state_manager.user_data.items():
                try:
                    state = EnhancedEmotionalState.from_dict(data)
                    avg = (state.favor + state.intimacy) / 2
                    averages.append((user_key, avg, state.favor, state.intimacy, state.attitude, state.relationship))
                except (TypeError, KeyError) as e:
                    logger.warning(f"用户 {user_key} 数据格式错误，跳过排行榜计算: {e}")
                    continue
        
        # 排序
        averages.sort(key=lambda x: x[1], reverse=reverse)
        
        # 转换为 RankingEntry 对象
        entries = []
        for i, (user_key, avg, favor, intimacy, attitude, relationship) in enumerate(averages[:limit], 1):
            display_name = self._format_user_display(user_key)
            entries.append(RankingEntry(
                rank=i,
                user_key=user_key,
                average_score=avg,
                favor=favor,
                intimacy=intimacy,
                attitude=attitude,
                relationship=relationship,
                display_name=display_name
            ))
        
        # 存入缓存
        await self.cache.set(cache_key, entries)
        
        return entries
    
    def _format_user_display(self, user_key: str) -> str:
        """格式化用户显示名称"""
        if '_' in user_key:
            try:
                session_id, user_id = user_key.split('_', 1)
                return f"用户{user_id}"
            except ValueError:
                return f"用户{user_key}"
        return f"用户{user_key}"


class EmotionAnalyzer:
    """情感分析器"""
    
    @classmethod
    def get_dominant_emotion(cls, state: EnhancedEmotionalState) -> str:
        """获取主导情感"""
        emotions = {
            "喜悦": state.joy,
            "信任": state.trust,
            "恐惧": state.fear,
            "惊讶": state.surprise,
            "悲伤": state.sadness,
            "厌恶": state.disgust,
            "愤怒": state.anger,
            "期待": state.anticipation
        }
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant[0] if dominant[1] > 0 else "中立"
    
    @classmethod
    def get_emotional_profile(cls, state: EnhancedEmotionalState, favor_weight: float = None, intimacy_weight: float = None) -> Dict[str, Any]:
        """获取完整的情感档案（支持动态权重）"""
        # 如果不提供权重，使用默认值
        if favor_weight is None:
            favor_weight = 0.6
        if intimacy_weight is None:
            intimacy_weight = 0.4
            
        dominant_emotion = cls.get_dominant_emotion(state)
        
        # 计算情感强度
        total_emotion = sum([
            state.joy, state.trust, state.fear, state.surprise,
            state.sadness, state.disgust, state.anger, state.anticipation
        ])
        emotion_intensity = min(100, total_emotion // 2)
        
        # 使用动态权重计算复合评分
        composite_score = state.favor * favor_weight + state.intimacy * intimacy_weight
        
        # 判断关系趋势
        if state.favor * favor_weight > state.intimacy * intimacy_weight:
            relationship_trend = "好感领先"
        elif state.intimacy * intimacy_weight > state.favor * favor_weight:
            relationship_trend = "亲密度领先" 
        else:
            relationship_trend = "平衡发展"
            
        # 计算互动质量
        total_interactions = state.interaction_count
        if total_interactions > 0:
            positive_ratio = (state.positive_interactions / total_interactions) * 100
        else:
            positive_ratio = 0
            
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_intensity": emotion_intensity,
            "relationship_trend": relationship_trend,
            "positive_ratio": positive_ratio,
            "composite_score": composite_score,
            "favor_weight": favor_weight,
            "intimacy_weight": intimacy_weight
        }


# ==================== 命令处理器类 ====================

class UserCommandHandler:
    """用户命令处理器"""
    
    def __init__(self, plugin):
        self.plugin = plugin
    
    async def show_emotional_state(self, event: AstrMessageEvent):
        """显示情感状态"""
        user_key = self.plugin._get_user_key(event)
        state = await self.plugin.user_manager.get_user_state(user_key)
    
        # 获取基础情感状态文本
        response_text = self.plugin._format_emotional_state(state)
    
        # 在回复开头添加用户标识（群聊和私聊都显示）
        if '_' in user_key:
            try:
                session_id, user_id = user_key.split('_', 1)
                user_identifier = f"群聊{session_id}_用户{user_id}"
            except ValueError:
                user_identifier = f"用户{user_key}"
        else:
            user_identifier = f"用户{user_key}"
    
        # 在原有回复前添加用户标识
        response_text = f"{user_identifier}\n\n{response_text}"
    
        yield event.plain_result(response_text)
        event.stop_event()
    
    async def toggle_status_display(self, event: AstrMessageEvent):
        """切换状态显示开关"""
        user_key = self.plugin._get_user_key(event)
        state = await self.plugin.user_manager.get_user_state(user_key)
        state.show_status = not state.show_status
        await self.plugin.user_manager.update_user_state(user_key, state)
        
        status_text = "开启" if state.show_status else "关闭"
        yield event.plain_result(f"【状态显示】已{status_text}状态显示")
        event.stop_event()
    
    async def show_favor_ranking(self, event: AstrMessageEvent, num: str = "10"):
        """显示好感度排行榜"""
        try:
            limit = min(int(num), 20)
            if limit <= 0:
                raise ValueError
        except ValueError:
            yield event.plain_result("【错误】排行数量必须是一个正整数（最大20）。")
            event.stop_event()
            return

        rankings = await self.plugin.ranking_manager.get_enhanced_ranking(limit, True)
        
        if not rankings:
            yield event.plain_result("【排行榜】当前没有任何用户数据。")
            event.stop_event()
            return

        response_lines = [f"【情感状态 TOP {limit} 排行榜】", "=================="]
        for entry in rankings:
            trend = "↑" if entry.average_score > 0 else "↓"
            line = (
                f"{entry.rank}. {entry.display_name}\n"
                f"   综合: {entry.average_score:.1f} {trend} | 态度: {entry.attitude} | 关系: {entry.relationship}\n"
                f"   好感: {entry.favor} | 亲密: {entry.intimacy}"
            )
            response_lines.append(line)
        
        yield event.plain_result("\n".join(response_lines))
        event.stop_event()
    
    async def show_negative_favor_ranking(self, event: AstrMessageEvent, num: str = "10"):
        """显示负好感排行榜"""
        try:
            limit = min(int(num), 20)
            if limit <= 0:
                raise ValueError
        except ValueError:
            yield event.plain_result("【错误】排行数量必须是一个正整数（最大20）。")
            event.stop_event()
            return

        rankings = await self.plugin.ranking_manager.get_enhanced_ranking(limit, False)
        
        if not rankings:
            yield event.plain_result("【排行榜】当前没有任何用户数据。")
            event.stop_event()
            return

        response_lines = [f"【情感状态 BOTTOM {limit} 排行榜】", "=================="]
        for entry in rankings:
            line = (
                f"{entry.rank}. {entry.display_name}\n"
                f"   综合: {entry.average_score:.1f} | 态度: {entry.attitude} | 关系: {entry.relationship}\n"
                f"   好感: {entry.favor} | 亲密: {entry.intimacy}"
            )
            response_lines.append(line)
        
        yield event.plain_result("\n".join(response_lines))
        event.stop_event()
    
    async def show_relationship_stage(self, event: AstrMessageEvent):
        """显示关系阶段详情"""
        user_key = self.plugin._get_user_key(event)
        state = await self.plugin.user_manager.get_user_state(user_key)
        
        stage_info = self.plugin.weight_manager.get_stage_info(state)
        stage_advice = self.plugin.weight_manager.get_stage_progression_advice(state)
        
        response_lines = [
            "【关系发展阶段分析】",
            "==================",
            f"当前阶段：{stage_info['stage_name']}",
            f"阶段描述：{stage_info['description']}",
            f"动态权重：好感度 {stage_info['favor_weight']*100:.0f}% | 亲密度 {stage_info['intimacy_weight']*100:.0f}%",
            f"复合评分：{stage_info['composite_score']:.1f} / {stage_info['next_stage_threshold']}",
            f"阶段进度：{stage_info['progress_to_next']:.1f}%",
        ]
        
        # 添加过渡状态信息
        if stage_info['is_transitioning']:
            if stage_info['intimacy_boost_active']:
                response_lines.extend([
                    "",
                    "【阶段过渡状态】",
                    f"过渡进度：{stage_info['transition_progress']:.1f}%",
                    f"需要亲密度提升：{stage_info['needed_intimacy_boost']}点",
                    f"状态：正在适应新阶段"
                ])
            else:
                response_lines.extend([
                    "",
                    "【阶段过渡状态】",
                    f"状态：过渡完成，关系已稳定"
                ])
        
        response_lines.extend([
            "",
            "【阶段进阶建议】",
            stage_advice,
            "",
            "【各阶段权重变化】",
            "初识期：好感70% | 亲密30% (建立吸引)",
            "深化期：好感50% | 亲密50% (共同成长)", 
            "承诺期：好感30% | 亲密70% (根基稳固)",
            "共生期：完全融合 (爱的终极形态)"
        ])
        
        yield event.plain_result("\n".join(response_lines))
        event.stop_event()


class AdminCommandHandler:
    """管理员命令处理器"""
    
    def __init__(self, plugin):
        self.plugin = plugin
    
    def _resolve_user_key(self, user_input: str) -> str:
        """解析用户输入的用户标识符"""
        if self.plugin.session_based:
            if '_' in user_input:
                return user_input
            else:
                for user_key in self.plugin.user_manager.user_data:
                    if user_key.endswith(f"_{user_input}"):
                        return user_key
                return user_input
        else:
            return user_input
    
    async def set_favor(self, event: AstrMessageEvent, user_input: str, value: str):
        """设置好感度"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
            
        try:
            favor_value = int(value)
            if not self.plugin.favour_min <= favor_value <= self.plugin.favour_max:
                yield event.plain_result(f"【错误】好感度值必须在 {self.plugin.favour_min} 到 {self.plugin.favour_max} 之间。")
                event.stop_event()
                return
        except ValueError:
            yield event.plain_result("【错误】好感度值必须是数字")
            event.stop_event()
            return
            
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        
        state = await self.plugin.user_manager.get_user_state(user_key)
        state.favor = favor_value
        
        await self.plugin.user_manager.update_user_state(user_key, state)
        await self.plugin.cache.set(f"state_{user_key}", state)
        
        mode_info = "（会话模式）" if self.plugin.session_based else ""
        yield event.plain_result(f"【成功】用户 {user_input}{mode_info} 的好感度已设置为 {favor_value}")
        event.stop_event()
    
    async def set_intimacy(self, event: AstrMessageEvent, user_input: str, value: str):
        """设置亲密度"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
            
        try:
            intimacy_value = int(value)
            if not self.plugin.intimacy_min <= intimacy_value <= self.plugin.intimacy_max:
                yield event.plain_result(f"【错误】亲密度值必须在 {self.plugin.intimacy_min} 到 {self.plugin.intimacy_max} 之间。")
                event.stop_event()
                return
        except ValueError:
            yield event.plain_result("【错误】亲密度值必须是数字")
            event.stop_event()
            return
            
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        
        state = await self.plugin.user_manager.get_user_state(user_key)
        state.intimacy = intimacy_value
        
        await self.plugin.user_manager.update_user_state(user_key, state)
        await self.plugin.cache.set(f"state_{user_key}", state)
        
        mode_info = "（会话模式）" if self.plugin.session_based else ""
        yield event.plain_result(f"【成功】用户 {user_input}{mode_info} 的亲密度已设置为 {intimacy_value}")
        event.stop_event()
    
    async def set_attitude(self, event: AstrMessageEvent, user_input: str, attitude: str):
        """设置态度"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
            
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        
        state = await self.plugin.user_manager.get_user_state(user_key)
        state.attitude = attitude
        
        await self.plugin.user_manager.update_user_state(user_key, state)
        await self.plugin.cache.set(f"state_{user_key}", state)
        
        mode_info = "（会话模式）" if self.plugin.session_based else ""
        yield event.plain_result(f"【成功】用户 {user_input}{mode_info} 的态度已设置为 {attitude}")
        event.stop_event()
    
    async def set_relationship(self, event: AstrMessageEvent, user_input: str, relationship: str):
        """设置关系"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
            
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        
        state = await self.plugin.user_manager.get_user_state(user_key)
        state.relationship = relationship
        
        await self.plugin.user_manager.update_user_state(user_key, state)
        await self.plugin.cache.set(f"state_{user_key}", state)
        
        mode_info = "（会话模式）" if self.plugin.session_based else ""
        yield event.plain_result(f"【成功】用户 {user_input}{mode_info} 的关系已设置为 {relationship}")
        event.stop_event()
    
    async def set_global_privacy_level(self, event: AstrMessageEvent, level: str):
        """设置全局隐私级别"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
        
        try:
            privacy_level = int(level)
            if not 0 <= privacy_level <= 2:
                raise ValueError
        except ValueError:
            yield event.plain_result("【错误】隐私级别必须是 0、1 或 2\n0=完全保密, 1=基础显示, 2=详细显示")
            event.stop_event()
            return
        
        # 更新全局配置
        self.plugin.global_privacy_level = privacy_level
        level_names = {0: "完全保密", 1: "基础显示", 2: "详细显示"}
        
        logger.info(f"管理员更新全局隐私级别: {level_names[privacy_level]}")
        yield event.plain_result(f"【全局设置】隐私级别已设置为: {level_names[privacy_level]}（全员生效）")
        event.stop_event()
    
    async def reset_favor(self, event: AstrMessageEvent, user_input: str):
        """重置用户好感度状态"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
            
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        new_state = EnhancedEmotionalState()
        
        await self.plugin.user_manager.update_user_state(user_key, new_state)
        await self.plugin.cache.set(f"state_{user_key}", new_state)
        
        mode_info = "（会话模式）" if self.plugin.session_based else ""
        yield event.plain_result(f"【成功】用户 {user_input}{mode_info} 的情感状态已完全重置")
        event.stop_event()
    
    async def view_favor(self, event: AstrMessageEvent, user_input: str):
        """管理员查看指定用户的好感状态"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
        
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        state = await self.plugin.user_manager.get_user_state(user_key)
        
        # 使用动态权重计算情感档案
        stage_info = self.plugin.weight_manager.get_stage_info(state)
        profile = self.plugin.analyzer.get_emotional_profile(state, stage_info['favor_weight'], stage_info['intimacy_weight'])
        
        # 格式化显示名称
        display_name = self.plugin.ranking_manager._format_user_display(user_input)
        
        response_lines = [
            f"【用户 {display_name} 完整情感状态】",
            f"用户标识: {user_key}",
            "==================",
            f"关系阶段: {stage_info['stage_name']} (进度: {stage_info['progress_to_next']:.1f}%)",
            f"动态权重: 好感{stage_info['favor_weight']*100:.0f}% | 亲密{stage_info['intimacy_weight']*100:.0f}%",
            f"态度: {state.attitude} | 关系: {state.relationship}",
            f"好感度: {state.favor} | 亲密度: {state.intimacy}",
            f"复合评分: {profile['composite_score']:.1f}",
            f"关系深度: {state.relationship_depth} | 态度强度: {state.attitude_intensity}",
            f"主导情感: {profile['dominant_emotion']} | 情感强度: {profile['emotion_intensity']}%",
            f"互动统计: {state.interaction_count}次 (正面: {state.positive_interactions}, 负面: {state.negative_interactions})",
            f"最后互动: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state.last_interaction)) if state.last_interaction > 0 else '从未互动'}",
            f"状态显示: {'开启' if state.show_status else '关闭'}",
        ]
        
        # 添加过渡状态信息
        if stage_info['is_transitioning']:
            if stage_info['intimacy_boost_active']:
                response_lines.extend([
                    f"过渡状态: 进行中 ({stage_info['transition_progress']:.1f}%)",
                    f"需要亲密度: +{stage_info['needed_intimacy_boost']}点"
                ])
            else:
                response_lines.append("过渡状态: 已完成")
        
        response_lines.extend([
            "",
            "【情感维度详情】",
            f"  喜悦: {state.joy} | 信任: {state.trust} | 恐惧: {state.fear} | 惊讶: {state.surprise}",
            f"  悲伤: {state.sadness} | 厌恶: {state.disgust} | 愤怒: {state.anger} | 期待: {state.anticipation}"
        ])
        
        yield event.plain_result("\n".join(response_lines))
        event.stop_event()
    
    async def reset_plugin(self, event: AstrMessageEvent):
        """重置插件所有数据"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        # 重置所有数据
        await self.plugin.user_manager.clear_all_data()
        await self.plugin.cache.clear()
        
        logger.info("管理员执行了插件重置操作")
        
        yield event.plain_result("【成功】插件所有数据已重置")
        event.stop_event()
    
    async def backup_data(self, event: AstrMessageEvent):
        """备份插件数据"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        try:
            backup_path = self.plugin._create_backup()
            yield event.plain_result(f"【成功】数据备份成功: {backup_path}")
            event.stop_event()
        except Exception as e:
            yield event.plain_result(f"【错误】备份失败: {str(e)}")
            event.stop_event()

# ==================== 情感分析专家类 ====================

class EmotionAnalysisExpert:
    """情感分析专家 - 专门处理关系和态度更新"""
    
    def __init__(self, plugin):
        self.plugin = plugin
        self.context = plugin.context
        self.cache = TTLCache(default_ttl=600, max_size=200)
    
    async def analyze_and_update_emotion(self, user_key: str, user_message: str, ai_response: str, 
                                       current_state: EnhancedEmotionalState) -> Dict[str, Any]:
        """分析对话并更新情感状态"""
    
        # 检查缓存
        cache_key = f"emotion_analysis_{user_key}_{hash(user_message)}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.info(f"使用缓存的情感分析结果")
            return cached_result
    
        # 构建情感分析提示词
        prompt = self._build_emotion_analysis_prompt(user_message, ai_response, current_state)
    
        # 尝试调用辅助 LLM，但如果失败则立即使用后备机制
        analysis_result = await self._call_secondary_llm(prompt)
    
        if analysis_result:
            # 解析分析结果
            updates = self._parse_emotion_analysis(analysis_result, current_state)
        
            # 缓存结果
            await self.cache.set(cache_key, updates)
        
            logger.info(f"情感分析完成: {updates}")
            return updates
        else:
            # 如果辅助 LLM 调用失败，直接使用后备更新
            logger.info("辅助LLM调用失败，使用后备更新")
            return self._generate_fallback_updates(user_message, ai_response, current_state)
    
    def _build_emotion_analysis_prompt(self, user_message: str, ai_response: str, 
                                     state: EnhancedEmotionalState) -> str:
        """构建情感分析提示词"""
        
        stage_info = self.plugin.weight_manager.get_stage_info(state)
        profile = self.plugin.analyzer.get_emotional_profile(state, stage_info['favor_weight'], stage_info['intimacy_weight'])
        
        return f"""
你是一个专业的情感分析专家，请根据对话内容分析情感状态并更新关系和态度描述。

【分析任务】
根据以下对话内容，分析情感变化并更新关系和态度描述。

【当前情感状态】
- 关系阶段: {stage_info['stage_name']} ({stage_info['description']})
- 当前关系: {state.relationship}
- 当前态度: {state.attitude} 
- 好感度: {state.favor}/100
- 亲密度: {state.intimacy}/100
- 复合评分: {stage_info['composite_score']:.1f}
- 主导情感: {profile['dominant_emotion']}
- 互动次数: {state.interaction_count}

【对话内容】
用户: {user_message}
AI: {ai_response}

【分析要求】
1. 分析对话的情感基调（正面/负面/中性）
2. 评估关系可能发生的变化
3. 评估态度可能发生的变化  
4. 考虑上下文连贯性（渐进式变化）

【输出格式】
请严格按照以下JSON格式输出：

{{
    "analysis": "对对话情感的分析描述",
    "relationship_change": "关系变化描述", 
    "attitude_change": "态度变化描述",
    "relationship_suggestion": "建议的新关系描述",
    "attitude_suggestion": "建议的新态度描述",
    "confidence": 0.8,
    "reasoning": "变化理由"
}}

【输出规则】
1. 关系和态度变化必须是渐进式的
2. 新描述要自然、符合人格设定
3. 变化幅度要合理，避免跳跃
4. 必须基于对话内容进行推理
"""
    
    async def _call_secondary_llm(self, prompt: str) -> str:
        """调用辅助LLM进行情感分析 - 修复版本"""
        try:
            logger.info("=== 辅助LLM调用开始 ===")
            providers = self.context.get_all_providers()
            if not providers:
                logger.error("没有可用的LLM提供商")
                return ""

            provider = providers[0]
            logger.info(f"使用提供商: {type(provider).__name__}")
            logger.info(f"使用模型: {self.plugin.config.get('secondary_llm_model', '默认模型')}")

            # ✅ 明确调用 text_chat 方法（ProviderOpenAIOfficial 支持）
            if hasattr(provider, 'text_chat') and asyncio.iscoroutinefunction(provider.text_chat):
                from astrbot.api.provider import ProviderRequest
                req = ProviderRequest(prompt=prompt)
                req.model = self.plugin.config.get('secondary_llm_model', None)

                logger.info("调用 provider.text_chat(prompt)")
                result = await provider.text_chat(prompt)  # 部分提供商支持直接传 prompt
                text = self._extract_response_text(result)
                if text:
                    logger.info("✅ 辅助LLM调用成功")
                    return text
                else:
                    logger.warning("❌ 返回内容为空或格式异常")

            # ✅ 备选：使用 text_chat_stream（同步方法，但支持流式）
            if hasattr(provider, 'text_chat_stream'):
                logger.info("尝试 text_chat_stream")
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(provider.text_chat_stream, prompt)
                    result = future.result(timeout=10)
                    text = self._extract_response_text(result)
                    if text:
                        logger.info("✅ text_chat_stream 成功")
                        return text

        except Exception as e:
            logger.error(f"辅助LLM调用异常: {e}")
            import traceback
            logger.error(f"异常堆栈: {traceback.format_exc()}")

        logger.error("❌ 所有辅助LLM调用方法均失败")
        return ""
    
    async def _call_secondary_llm_with_retry(self, prompt: str) -> str:
        for i in range(3):
            try:
                return await self._call_secondary_llm(prompt)
            except Exception as e:
                if i == 2:
                    raise
                await asyncio.sleep(1)
    
    def _extract_response_text(self, response_obj) -> str:
        """从响应对象中提取文本"""
        if response_obj is None:
            return ""
    
        # 尝试不同的属性名
        for attr in ['completion_text', 'text', 'content', 'response', 'result', 'message']:
            if hasattr(response_obj, attr):
                text = getattr(response_obj, attr)
                if text and isinstance(text, str):
                    return text
    
        # 如果是字符串，直接返回
        if isinstance(response_obj, str):
            return response_obj
    
        # 如果是字典，尝试提取文本
        if isinstance(response_obj, dict):
            for key in ['completion_text', 'text', 'content', 'response', 'result', 'message']:
                if key in response_obj and isinstance(response_obj[key], str):
                    return response_obj[key]
    
        # 尝试转换为字符串
        try:
            text = str(response_obj)
            if text and text not in ['None', '']:
                return text
        except:
            pass
    
        return ""
    
    def _parse_emotion_analysis(self, analysis_text: str, current_state: EnhancedEmotionalState) -> Dict[str, Any]:
        """解析情感分析结果"""
        import json
        import re
        
        updates = {}
        
        try:
            # 尝试提取JSON格式
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
                
                # 应用建议的更新
                if analysis_data.get("relationship_suggestion"):
                    updates['relationship_text'] = analysis_data["relationship_suggestion"]
                
                if analysis_data.get("attitude_suggestion"):
                    updates['attitude_text'] = analysis_data["attitude_suggestion"]
                
                # 根据分析置信度调整情感数值
                confidence = analysis_data.get("confidence", 0.5)
                reasoning = analysis_data.get("reasoning", "").lower()
                
                # 基于分析结果调整基础情感
                if "正面" in reasoning or "积极" in reasoning:
                    updates.update({
                        'joy': max(1, int(2 * confidence)),
                        'trust': max(1, int(1 * confidence)),
                        'favor': max(1, int(1 * confidence)),
                        'intimacy': max(1, int(1 * confidence))
                    })
                elif "负面" in reasoning or "消极" in reasoning:
                    updates.update({
                        'sadness': max(1, int(2 * confidence)),
                        'anger': max(1, int(1 * confidence)),
                        'favor': min(-1, int(-1 * confidence)),
                        'intimacy': min(-1, int(-1 * confidence))
                    })
                else:
                    # 中性
                    updates.update({
                        'trust': 1,
                        'intimacy': 1
                    })
                
                updates['source'] = 'emotion_expert'
                updates['analysis_data'] = analysis_data
                
                logger.info(f"情感分析解析成功")
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"情感分析JSON解析失败: {e}")
            updates.update(self._extract_updates_from_text(analysis_text))
        
        return updates
    
    def _extract_updates_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取更新信息"""
        updates = {}
        
        text_lower = text.lower()
        
        # 简单关键词匹配
        relationship_keywords = {
            "亲近": "稍微更亲近", "熟悉": "更加熟悉", "信任": "增加信任",
            "疏远": "略显疏远", "陌生": "保持距离", "友好": "更加友好"
        }
        
        for keyword, suggestion in relationship_keywords.items():
            if keyword in text_lower:
                updates['relationship_text'] = suggestion
                break
        
        attitude_keywords = {
            "开心": "愉快交流", "积极": "积极回应", "热情": "热情对话",
            "谨慎": "谨慎回应", "严肃": "严肃对待", "冷淡": "稍显冷淡"
        }
        
        for keyword, suggestion in attitude_keywords.items():
            if keyword in text_lower:
                updates['attitude_text'] = suggestion
                break
        
        if updates:
            updates['source'] = 'text_analysis'
        
        return updates
    
    def _generate_fallback_updates(self, user_message: str, ai_response: str, state: EnhancedEmotionalState) -> Dict[str, Any]:
        response_lower = ai_response.lower()

        # 情感关键词库（可扩展）
        positive_words = ['开心', '高兴', '喜欢', '爱', '谢谢', '好', '棒', '可爱', '想', '抱']
        negative_words = ['讨厌', '恨', '生气', '愤怒', '伤心', '难过', '抱歉', '对不起', '不', '烦']

        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)

        # 模板库
        if positive_count > negative_count:
            attitude_templates = ["愉快回应", "温柔回应", "积极互动", "带着笑意回应"]
            relationship_templates = ["逐渐亲近", "建立联系", "互相了解", "开始信任"]
        elif negative_count > positive_count:
            attitude_templates = ["谨慎回应", "稍显冷淡", "情绪低落", "保持距离"]
            relationship_templates = ["需要空间", "观察中", "略显疏远", "重新建立连接"]
        else:
            attitude_templates = ["平静回应", "自然交流", "温和对话", "照常回应"]
            relationship_templates = ["稳定交流", "平常关系", "持续互动", "熟悉中"]

        # 随机选择（避免重复）
        import random
        attitude = random.choice(attitude_templates)
        relationship = random.choice(relationship_templates)

        return {
            'attitude_text': attitude,
            'relationship_text': relationship,
            'source': 'enhanced_fallback'
        }

# ==================== 主插件类 ====================

@register("EmotionAI Pro", "融合版", "融合 EmotionAI 与 FavourPro 的高级情感智能交互系统", "3.3.0")
class EmotionAIProPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        # 配置验证和初始化
        self._validate_and_init_config()
        
        # 获取规范的数据目录
        data_dir = StarTools.get_data_dir() / "emotionai_pro"
        
        # 初始化各个管理器
        self.user_manager = UserStateManager(data_dir)
        self.ranking_manager = RankingManager(self.user_manager)
        self.analyzer = EmotionAnalyzer()
        self.attitude_manager = AttitudeRelationshipManager()
        self.weight_manager = DynamicWeightManager()
        
        # 缓存系统
        self.cache = TTLCache(default_ttl=300, max_size=500)
        
        # 情感分析专家（新增）
        self.emotion_expert = EmotionAnalysisExpert(self)
        
        # 命令处理器
        self.user_commands = UserCommandHandler(self)
        self.admin_commands = AdminCommandHandler(self)
        
        # 原有的正则表达式模式
        self.inner_assessment_pattern = re.compile(
            r"\[\s*内心评估:\s*Favour:\s*(-?\d+)\s*,\s*Attitude:\s*(.+?)\s*,\s*Relationship:\s*(.+?)\s*\]",
            re.DOTALL
        )
        self.favourpro_pattern = re.compile(
            r"\[\s*Favour:\s*(-?\d+)\s*,\s*Attitude:\s*(.+?)\s*,\s*Relationship:\s*(.+?)\s*\]",
            re.DOTALL
        )
        self.emotionai_pattern = re.compile(r"\[情感更新:\s*(.*?)\]", re.DOTALL)
        self.single_emotion_pattern = re.compile(r"(\w+):\s*([+-]?\d+)")
        
        # 自动保存任务
        self.auto_save_task = asyncio.create_task(self._auto_save_loop())
        
        logger.info(f"EmotionAI Pro 插件初始化完成 - 情感分析专家版本")
        logger.info(f"辅助LLM: {self.enable_secondary_llm}")
        
    def _validate_and_init_config(self):
        """验证配置并初始化配置参数"""
        # 基础配置
        self.session_based = bool(self.config.get("session_based", False))
        self.favour_min = self.config.get("favour_min", -100)
        self.favour_max = self.config.get("favour_max", 100)
        self.intimacy_min = self.config.get("intimacy_min", 0)
        self.intimacy_max = self.config.get("intimacy_max", 100)
        self.change_min = self.config.get("change_min", -10)
        self.change_max = self.config.get("change_max", 5)
        self.admin_qq_list = self.config.get("admin_qq_list", [])
        self.plugin_priority = self.config.get("plugin_priority", 100000)
    
        # 融合特性配置
        self.enable_attitude_system = self.config.get("enable_attitude_system", True)
        self.enable_ai_text_generation = self.config.get("enable_ai_text_generation", True)
        self.global_privacy_level = self.config.get("global_privacy_level", 1)
    
        # 动态权重配置
        self.enable_dynamic_weights = self.config.get("enable_dynamic_weights", True)
    
        # 情感分析专家配置（新增）
        self.enable_secondary_llm = self.config.get("enable_secondary_llm", True)
        self.secondary_llm_provider = self.config.get("secondary_llm_provider", "")
        self.secondary_llm_model = self.config.get("secondary_llm_model", "")
        self.force_text_update = self.config.get("force_text_update", True)
    
        logger.info(f"融合配置加载: 态度系统={self.enable_attitude_system}, "
                   f"AI文本生成={self.enable_ai_text_generation}, "
                   f"辅助LLM={self.enable_secondary_llm}")
        
    async def _auto_save_loop(self):
        """自动保存循环"""
        while True:
            try:
                await asyncio.sleep(30)
                await self.user_manager.force_save()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"自动保存失败: {e}")
                
    def _get_user_key(self, event: AstrMessageEvent) -> str:
        """获取用户键"""
        user_id = event.get_sender_id()
        if self.session_based:
            session_id = event.unified_msg_origin
            return f"{session_id}_{user_id}"
        return user_id
        
    def _get_session_id(self, event: AstrMessageEvent) -> Optional[str]:
        """获取会话ID"""
        return event.unified_msg_origin if self.session_based else None
    
    def _get_message_text(self, event: AstrMessageEvent) -> str:
        """获取消息文本的兼容性方法"""
        try:
            # 方法1: 尝试使用 get_plain_text
            if hasattr(event, 'get_plain_text'):
                return event.get_plain_text()
            
            # 方法2: 尝试使用 message 属性
            if hasattr(event, 'message'):
                message = event.message
                if isinstance(message, str):
                    return message
                # 如果是消息链，提取文本
                elif hasattr(message, 'extract_plain_text'):
                    return message.extract_plain_text()
            
            # 方法3: 尝试使用 raw_message 属性
            if hasattr(event, 'raw_message'):
                return event.raw_message
            
            # 方法4: 最后尝试转换为字符串
            return str(event)
            
        except Exception as e:
            logger.warning(f"获取消息文本失败: {e}")
            return ""    
        
    def _format_emotional_state(self, state: EnhancedEmotionalState) -> str:
        """格式化情感状态显示（包含动态权重信息）"""
        if self.global_privacy_level == 0:
            return "【情感状态】*保密*"
    
        # 获取阶段信息
        stage_info = self.weight_manager.get_stage_info(state)
        stage_advice = self.weight_manager.get_stage_progression_advice(state)
    
        # 更新状态的阶段信息
        state.update_stage_info(stage_info)
    
        # 计算复合评分
        composite_score = stage_info['composite_score']
    
        if self.global_privacy_level == 1:
            # 确保进度显示不为负数
            progress_display = max(0, stage_info['progress_to_next'])
        
            base_info = (
                "【当前情感状态】\n"
                "====================================\n"
                f"关系阶段：{stage_info['stage_name']} ({progress_display:.1f}%)\n"  # 使用修正后的进度
                f"复合评分：{composite_score:.1f}\n"
                f"关系：{state.relationship}\n"
                f"态度：{state.attitude}"
            )
        
            # 添加过渡状态提示
            if stage_info['is_transitioning']:
                if stage_info['intimacy_boost_active']:
                    base_info += f"\n过渡期: 需要提升亲密度 {stage_info['needed_intimacy_boost']}点"
                else:
                    base_info += f"\n过渡完成"
                
            return base_info
    
        else:  # 详细显示
            profile = self.analyzer.get_emotional_profile(state, stage_info['favor_weight'], stage_info['intimacy_weight'])
            frequency = self._get_interaction_frequency(state)
        
            # 确保进度显示不为负数
            progress_display = max(0, stage_info['progress_to_next'])
        
            detailed_info = (
                "【当前情感状态】\n"
                "====================================\n"
                f"关系阶段：{stage_info['stage_name']}\n"
                f"   {stage_info['description']}\n"
            )
        
            # 添加过渡状态信息
            if stage_info['is_transitioning']:
                if stage_info['intimacy_boost_active']:
                    detailed_info += (
                        f"   阶段过渡中 ({stage_info['transition_progress']:.1f}%)\n"
                        f"   需要亲密度提升: +{stage_info['needed_intimacy_boost']}点\n"
                    )
                else:
                    detailed_info += f"   过渡完成\n"
            else:
                # 使用修正后的进度
                detailed_info += f"   阶段进度：{progress_display:.1f}% (下一阶段: {stage_info['next_stage_threshold']}+)\n"
        
            # 如果是负好感，显示特殊的权重信息
            if state.favor < 0:
                weight_info = "   好感度：100% | 亲密度：0% (负好感模式)\n"
            else:
                weight_info = f"   好感度：{stage_info['favor_weight']*100:.0f}% | 亲密度：{stage_info['intimacy_weight']*100:.0f}%\n"
        
            detailed_info += (
                f"\n动态权重\n"
                f"{weight_info}"
                f"   复合评分：{stage_info['composite_score']:.1f}\n\n"
                f"核心状态\n"
                f"   关系：{state.relationship} | 态度：{state.attitude}\n"
                f"   好感度：{state.favor} | 亲密度：{state.intimacy}\n"
                f"   主导情感：{profile['dominant_emotion']} | 趋势：{profile['relationship_trend']}\n\n"
                f"互动统计\n"
                f"   次数：{state.interaction_count}次 ({frequency})\n"
                f"   正面互动：{profile['positive_ratio']:.1f}%\n\n"
                f"阶段建议\n"
                f"   {stage_advice}\n\n"
                f"情感维度\n"
                f"   喜悦：{state.joy} | 信任：{state.trust} | 恐惧：{state.fear} | 惊讶：{state.surprise}\n"
                f"   悲伤：{state.sadness} | 厌恶：{state.disgust} | 愤怒：{state.anger} | 期待：{state.anticipation}"
            )
        
            return detailed_info
            
    def _format_time(self, timestamp: float) -> str:
        """格式化时间"""
        if timestamp == 0:
            return "从未互动"
        return time.strftime("%m-%d %H:%M", time.localtime(timestamp))
            
    def _get_interaction_frequency(self, state: EnhancedEmotionalState) -> str:
        """获取互动频率描述"""
        if state.interaction_count == 0:
            return "首次互动"
            
        days_since_last = (time.time() - state.last_interaction) / (24 * 3600)
        if days_since_last < 1:
            return "频繁互动"
        elif days_since_last < 3:
            return "经常互动"
        elif days_since_last < 7:
            return "偶尔互动"
        else:
            return "稀少互动"
    
    # ==================== LLM集成 ====================
    
    @filter.on_llm_request(priority=100000)
    async def inject_enhanced_context(self, event: AstrMessageEvent, req: ProviderRequest):
        """注入增强的情感上下文"""
        user_key = self._get_user_key(event)
        
        # 从缓存获取状态或从管理器获取
        state = await self.cache.get(f"state_{user_key}")
        if state is None:
            state = await self.user_manager.get_user_state(user_key)
            await self.cache.set(f"state_{user_key}", state)
        
        # 构建融合的情感上下文
        emotional_context = self._build_enhanced_context(state)
        req.system_prompt += f"\n{emotional_context}"
        
    def _build_enhanced_context(self, state: EnhancedEmotionalState) -> str:
        """构建融合两个系统的上下文提示词"""
    
        # 获取动态权重信息
        stage_info = self.weight_manager.get_stage_info(state)
        behavior_guidance = self.attitude_manager.get_behavior_guidance(state.favor, state.intimacy)
    
        # 检查是否需要强制更新文本描述
        force_text_update = getattr(self, 'force_text_update', True)
        should_update_text = state.should_update_text_descriptions(force_text_update)
    
        text_update_instruction = ""
        if should_update_text:
            text_update_instruction = """
    【强制文本描述更新 - 本次对话必须执行】
    根据当前对话内容和上下文，你必须重新评估并更新态度和关系描述。
    即使变化很小，也要反映最新的互动状态。

    文本更新指导：
    1. 态度描述：反映当前对话的情绪基调和即时感受
       - 示例："愉快交谈"、"严肃讨论"、"轻松玩笑"、"关切询问"
       - 要具体：不要只用"友好"、"中立"等泛泛之词

    2. 关系描述：体现关系的最新动态和发展
       - 示例："分享生活点滴的伙伴"、"信任的倾诉对象"、"互相支持的友人"
       - 要渐进：使用"稍微更亲近"、"略显疏远"、"更加信任"等渐进描述

    3. 上下文关联：必须结合当前对话的具体内容
       - 如果用户分享个人经历 → 关系可更新为"分享更多内心的朋友"
       - 如果对话涉及情感话题 → 态度可更新为"情感上更开放的交流"
       - 如果是轻松玩笑 → 态度可更新为"轻松愉快的互动氛围"

    【本次对话必须输出文本更新】"""
        else:
            text_update_instruction = """
    【可选文本描述更新】
    如果觉得态度或关系有明显变化，可以更新描述。"""

        return f"""
    【机密指令】这是一个记录你内心真实想法的秘密评估系统。请遵循以下规则，并在回复内容结束后，换行并严格按照指定格式更新情感状态。

    --- 核心规则 ---
    1. **好感度范围**: -100 (极度厌恶) 到 100 (挚爱)
    2. **亲密度范围**: 0 (无感) 到 100 (相互依赖)  
    3. **动态权重系统**: 关系发展阶段决定权重分配
    4. **复合评分**: 加权计算 = 好感度×好感权重 + 亲密度×亲密度权重

    【当前状态】
    关系阶段：{stage_info['stage_name']} - {stage_info['description']}
    动态权重：好感度 {stage_info['favor_weight']*100:.0f}% | 亲密度 {stage_info['intimacy_weight']*100:.0f}%
    当前关系：{state.relationship} | 态度：{state.attitude}
    好感度：{state.favor}/100 | 亲密度：{state.intimacy}/100
    复合评分：{stage_info['composite_score']:.1f}
    互动统计：{state.interaction_count}次

    【情感响应指导】
    {behavior_guidance}

    【情感更新机制 - 每次必须执行】
    在每次回复末尾使用以下格式更新：
    [情感更新: joy:1, trust:1, favor:1, intimacy:1]

    基础情感维度：joy, trust, fear, surprise, sadness, disgust, anger, anticipation
    关系状态维度：favor, intimacy

    {text_update_instruction}

    【关系和态度更新格式】
    使用以下格式更新文本描述（如果需要更新）：
    [内心评估: Favour: {state.favor}, Attitude: 新的态度描述, Relationship: 新的关系描述]

    更新规则：
    - 态度描述要反映当前对话的情绪基调
    - 关系描述要体现关系的最新动态  
    - 即使是微小变化也要如实记录
    - 必须与当前对话内容相关联

    【重要提示】
    - 情感更新必须反映真实感受，即使是微小变化(+1/-1)也值得记录
    - 关系和态度描述应与对话内容和数值等级保持逻辑一致
    - 情感是持续流动的，每次对话都可能带来变化
    """
    
    @filter.on_llm_response(priority=100000)
    async def process_enhanced_update(self, event: AstrMessageEvent, resp: LLMResponse):
        """处理增强的情感更新（集成情感分析专家）"""
        user_key = self._get_user_key(event)
        original_text = resp.completion_text
        user_message = self._get_message_text(event)  # 使用兼容性方法

        logger.info(f"[DEBUG] ==== 开始处理LLM响应 ====")
        logger.info(f"[DEBUG] 用户: {user_key}")
        logger.info(f"[DEBUG] 用户消息: '{user_message}'")
        logger.info(f"[DEBUG] AI回复内容: '{original_text}'")
    
        # 解析增强的更新格式
        updates = self._parse_enhanced_updates(original_text)

        logger.info(f"[DEBUG] 解析到的情感更新: {updates}")
    
        # 获取当前状态
        state = await self.user_manager.get_user_state(user_key)

        # 如果启用了情感分析专家，并且需要文本更新
        if self.enable_secondary_llm and self._should_use_emotion_expert(updates, state):
            logger.info("调用情感分析专家进行深度分析")
        
            # 调用情感分析专家
            expert_updates = await self.emotion_expert.analyze_and_update_emotion(
                user_key, user_message, original_text, state
            )
        
            # 合并更新（专家分析优先）
            if expert_updates:
                updates.update(expert_updates)
                logger.info(f"情感专家更新合并成功")

        # 如果仍然没有文本更新，使用强制更新
        if self.force_text_update and (not updates.get('attitude_text') or not updates.get('relationship_text')):
            logger.info("强制生成文本描述更新")
            forced_updates = self._generate_forced_text_updates(user_message, original_text, state)
            updates.update(forced_updates)

        # 应用情感更新
        self._apply_enhanced_updates(state, updates)

        logger.info(f"[DEBUG] 更新后状态 - 好感:{state.favor}, 亲密:{state.intimacy}")
        logger.info(f"[DEBUG] 更新后态度: '{state.attitude}', 关系: '{state.relationship}'")
        logger.info(f"[DEBUG] ==== LLM响应处理完成 ====")
    
        # 更新互动统计
        self._update_interaction_stats(state, updates)
    
        # 清理回复文本
        cleaned_text = self._clean_enhanced_response_text(original_text)
        resp.completion_text = cleaned_text
    
        # 保存状态
        await self.user_manager.update_user_state(user_key, state)
        await self.cache.set(f"state_{user_key}", state)
    
        # 记录详细的更新日志
        if updates:
            source = updates.get('source', 'unknown')
            logger.info(f"用户 {user_key} {source}更新: 好感{state.favor}, 亲密{state.intimacy}, 态度'{state.attitude}', 关系'{state.relationship}'")
    
        # 根据用户设置和全局隐私级别显示状态
        if state.show_status and updates and self.global_privacy_level > 0:
            status_text = self._format_emotional_state(state)
            resp.completion_text += f"\n\n{status_text}"

    def _parse_enhanced_updates(self, text: str) -> Dict[str, Any]:
        """解析增强的情感更新（支持强制文本更新）"""
        updates = {}
    
        # 格式一：内心评估格式 [内心评估: Favour: X, Attitude: 描述, Relationship: 描述]
        match = self.inner_assessment_pattern.search(text)
        if match:
            try:
                updates['favor'] = int(match.group(1))
                updates['attitude_text'] = match.group(2).strip()
                updates['relationship_text'] = match.group(3).strip()
                updates['source'] = 'inner_assessment'
                logger.info(f"解析到内心评估更新: 好感={updates['favor']}, 态度='{updates['attitude_text']}', 关系='{updates['relationship_text']}'")
            except (ValueError, IndexError) as e:
                logger.warning(f"内心评估格式解析失败: {e}")
    
        # 格式二：传统 FavourPro 格式
        if not match:
            match = self.favourpro_pattern.search(text)
            if match:
                try:
                    updates['favor'] = int(match.group(1))
                    updates['attitude_text'] = match.group(2).strip()
                    updates['relationship_text'] = match.group(3).strip()
                    updates['source'] = 'favourpro'
                    logger.info(f"解析到FavourPro格式更新: 好感={updates['favor']}, 态度='{updates['attitude_text']}', 关系='{updates['relationship_text']}'")
                except (ValueError, IndexError) as e:
                    logger.warning(f"FavourPro格式解析失败: {e}")
    
        # 格式三：EmotionAI 格式的情感更新
        emotion_match = self.emotionai_pattern.search(text)
        if emotion_match:
            emotion_content = emotion_match.group(1)
            single_matches = self.single_emotion_pattern.findall(emotion_content)
            for emotion, value in single_matches:
                try:
                    change_value = int(value)
                    if emotion.lower() == 'favor':
                        change_value = max(self.change_min, min(self.change_max, change_value))
                    updates[emotion.lower()] = change_value
                except ValueError:
                    continue
            if single_matches:
                updates['source'] = 'emotionai'
                logger.info(f"解析到EmotionAI格式更新: {updates}")
    
        # 强制文本更新检查
        force_text_update = getattr(self, 'force_text_update', True)
        if force_text_update and not ('attitude_text' in updates or 'relationship_text' in updates):
            # 如果有情感数值变化但缺少文本更新，标记需要文本更新
            has_emotion_change = any(attr in updates for attr in 
                                   ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation', 
                                    'favor', 'intimacy'])
            if has_emotion_change:
                updates['needs_text_update'] = True
                logger.info("强制文本更新：检测到情感变化但缺少文本描述")
    
        return updates

    def _apply_enhanced_updates(self, state: EnhancedEmotionalState, updates: Dict[str, Any]):
        """应用增强的情感更新"""
    
        # 在应用更新前，先应用过渡期增益
        updates = self.weight_manager.apply_transition_benefits(state, updates)
    
        # 应用数值更新
        emotion_attrs = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
        state_attrs = ['favor', 'intimacy']
    
        for attr in emotion_attrs:
            if attr in updates:
                new_value = getattr(state, attr) + updates[attr]
                setattr(state, attr, max(0, min(100, new_value)))
    
        for attr in state_attrs:
            if attr in updates:
                new_value = getattr(state, attr) + updates[attr]
                if attr == 'favor':
                    setattr(state, attr, max(self.favour_min, min(self.favour_max, new_value)))
                else:
                    setattr(state, attr, max(self.intimacy_min, min(self.intimacy_max, new_value)))
    
        current_time = time.time()
    
        # 应用 AI 自主文本描述
        if 'attitude_text' in updates and updates['attitude_text']:
            state.attitude = updates['attitude_text']
            state.last_attitude_update = current_time
            state.attitude_update_count += 1
            logger.info(f"更新态度描述: '{updates['attitude_text']}'")
    
        if 'relationship_text' in updates and updates['relationship_text']:
            state.relationship = updates['relationship_text']
            state.last_relationship_update = current_time
            state.relationship_update_count += 1
            logger.info(f"更新关系描述: '{updates['relationship_text']}'")
        
    def _clean_enhanced_response_text(self, text: str) -> str:
        """清理增强的回复文本"""
        # 清理内心评估格式
        text = self.inner_assessment_pattern.sub('', text)
        # 清理传统格式
        text = self.favourpro_pattern.sub('', text)
        # 清理 EmotionAI 格式
        text = self.emotionai_pattern.sub('', text)
        return text.strip()
        
    def _update_interaction_stats(self, state: EnhancedEmotionalState, updates: Dict[str, Any]):
        """更新互动统计"""
        state.interaction_count += 1
        state.last_interaction = time.time()
        
        # 判断互动性质
        if updates:
            positive_emotions = sum([
                updates.get('joy', 0),
                updates.get('trust', 0), 
                updates.get('surprise', 0),
                updates.get('anticipation', 0)
            ])
            negative_emotions = sum([
                updates.get('fear', 0),
                updates.get('sadness', 0),
                updates.get('disgust', 0),
                updates.get('anger', 0)
            ])
            
            if positive_emotions > negative_emotions:
                state.positive_interactions += 1
            elif negative_emotions > positive_emotions:
                state.negative_interactions += 1

    def _should_use_emotion_expert(self, updates: Dict[str, Any], state: EnhancedEmotionalState) -> bool:
        """判断是否需要使用情感分析专家"""
        # 如果没有文本更新，使用专家
        if not updates.get('attitude_text') or not updates.get('relationship_text'):
            return True
        
        # 检查更新时间间隔
        current_time = time.time()
        time_since_attitude = current_time - getattr(state, 'last_attitude_update', 0)
        time_since_relationship = current_time - getattr(state, 'last_relationship_update', 0)
        
        if time_since_attitude > 600 or time_since_relationship > 600:  # 10分钟
            return True
        
        return False
    
    def _generate_forced_text_updates(self, user_message: str, ai_response: str, 
                                    state: EnhancedEmotionalState) -> Dict[str, Any]:
        """生成强制文本更新"""
        updates = {}
        
        # 简单的情感倾向分析
        response_lower = ai_response.lower()
        
        positive_keywords = ['开心', '高兴', '喜欢', '爱', '谢谢', '感谢', '好', '棒', '可爱']
        negative_keywords = ['讨厌', '恨', '生气', '愤怒', '伤心', '难过', '抱歉', '对不起', '不']
        
        positive_count = sum(1 for word in positive_keywords if word in response_lower)
        negative_count = sum(1 for word in negative_keywords if word in response_lower)
        
        # 生成渐进式更新
        if positive_count > negative_count:
            attitude_options = ["愉快交流", "积极回应", "热情对话", "开心互动"]
            relationship_options = ["逐渐熟悉", "建立联系", "友好互动", "开始了解"]
        elif negative_count > positive_count:
            attitude_options = ["谨慎回应", "低沉情绪", "严肃对话", "保留态度"]
            relationship_options = ["保持距离", "需要时间", "重新评估", "观察中"]
        else:
            attitude_options = ["平常交流", "标准回应", "礼貌对话", "一般互动"]
            relationship_options = ["平常关系", "一般交往", "正常交流", "普通相识"]
        
        # 选择不同的选项避免重复
        attitude_update_count = getattr(state, 'attitude_update_count', 0)
        relationship_update_count = getattr(state, 'relationship_update_count', 0)
        
        attitude_index = attitude_update_count % len(attitude_options)
        relationship_index = relationship_update_count % len(relationship_options)
        
        updates['attitude_text'] = attitude_options[attitude_index]
        updates['relationship_text'] = relationship_options[relationship_index]
        updates['source'] = 'forced_update'
        
        return updates            
    
    # ==================== 用户命令 ====================
    
    @filter.command("好感度", priority=5)
    async def show_emotional_state(self, event: AstrMessageEvent):
        """显示情感状态"""
        async for result in self.user_commands.show_emotional_state(event):
            yield result
        
    @filter.command("状态显示", priority=5)
    async def toggle_status_display(self, event: AstrMessageEvent):
        """切换状态显示开关"""
        async for result in self.user_commands.toggle_status_display(event):
            yield result
        
    @filter.command("关系阶段", priority=5)
    async def show_relationship_stage(self, event: AstrMessageEvent):
        """显示关系阶段详情"""
        async for result in self.user_commands.show_relationship_stage(event):
            yield result
        
    # ==================== 排行榜命令 ====================
    
    @filter.command("好感排行", priority=5)
    async def show_favor_ranking(self, event: AstrMessageEvent, num: str = "10"):
        """显示好感度排行榜"""
        async for result in self.user_commands.show_favor_ranking(event, num):
            yield result
        
    @filter.command("负好感排行", priority=5)
    async def show_negative_favor_ranking(self, event: AstrMessageEvent, num: str = "10"):
        """显示负好感排行榜"""
        async for result in self.user_commands.show_negative_favor_ranking(event, num):
            yield result
        
    # ==================== 缓存统计命令 ====================
    
    @filter.command("缓存统计", priority=5)
    async def show_cache_stats(self, event: AstrMessageEvent):
        """显示缓存统计信息"""
        stats = await self.cache.get_stats()
        
        response = [
            "【缓存统计信息】",
            "==================",
            f"缓存条目: {stats['total_entries']}",
            f"访问次数: {stats['access_count']}",
            f"命中次数: {stats['hit_count']}",
            f"命中率: {stats['hit_rate']}%",
            f"",
            f"提示: 缓存用于提高情感状态读取性能"
        ]
        
        yield event.plain_result("\n".join(response))
        event.stop_event()
        
    # ==================== 管理员命令 ====================
    
    def _is_admin(self, event: AstrMessageEvent) -> bool:
        """检查管理员权限"""
        return event.role == "admin" or event.get_sender_id() in self.admin_qq_list
        
    @filter.command("设置好感", priority=5)
    async def admin_set_favor(self, event: AstrMessageEvent, user_input: str, value: str):
        """设置好感度"""
        async for result in self.admin_commands.set_favor(event, user_input, value):
            yield result
        
    @filter.command("设置亲密", priority=5)
    async def admin_set_intimacy(self, event: AstrMessageEvent, user_input: str, value: str):
        """设置亲密度"""
        async for result in self.admin_commands.set_intimacy(event, user_input, value):
            yield result
        
    @filter.command("设置态度", priority=5)
    async def admin_set_attitude(self, event: AstrMessageEvent, user_input: str, attitude: str):
        """设置态度"""
        async for result in self.admin_commands.set_attitude(event, user_input, attitude):
            yield result
        
    @filter.command("设置关系", priority=5)
    async def admin_set_relationship(self, event: AstrMessageEvent, user_input: str, relationship: str):
        """设置关系"""
        async for result in self.admin_commands.set_relationship(event, user_input, relationship):
            yield result
        
    @filter.command("隐私级别", priority=5)
    async def admin_set_privacy_level(self, event: AstrMessageEvent, level: str):
        """设置全局隐私级别"""
        async for result in self.admin_commands.set_global_privacy_level(event, level):
            yield result
        
    @filter.command("重置好感", priority=5)
    async def admin_reset_favor(self, event: AstrMessageEvent, user_input: str):
        """重置用户好感度状态"""
        async for result in self.admin_commands.reset_favor(event, user_input):
            yield result
        
    @filter.command("重置插件", priority=5)
    async def admin_reset_plugin(self, event: AstrMessageEvent):
        """重置插件所有数据"""
        async for result in self.admin_commands.reset_plugin(event):
            yield result
    
    @filter.command("查看好感", priority=5)
    async def admin_view_favor(self, event: AstrMessageEvent, user_input: str):
        """管理员查看指定用户的好感状态"""
        async for result in self.admin_commands.view_favor(event, user_input):
            yield result
        
    @filter.command("备份数据", priority=5)
    async def admin_backup_data(self, event: AstrMessageEvent):
        """备份插件数据"""
        async for result in self.admin_commands.backup_data(event):
            yield result
            
    def _create_backup(self) -> str:
        """创建数据备份"""
        data_dir = StarTools.get_data_dir() / "emotionai_pro"
        backup_dir = data_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        backup_name = f"emotionai_pro_backup_{timestamp}"
        backup_path = backup_dir / backup_name
        
        # 创建备份目录
        backup_path.mkdir(exist_ok=True)
        
        # 复制数据文件
        for filename in ["user_emotion_data.json"]:
            src = data_dir / filename
            if src.exists():
                dst = backup_path / filename
                shutil.copy2(src, dst)
        
        return str(backup_path.relative_to(data_dir))
        
    async def terminate(self):
        """插件终止时清理资源"""
        if hasattr(self, 'auto_save_task'):
            self.auto_save_task.cancel()
            try:
                await self.auto_save_task
            except asyncio.CancelledError:
                pass
                
        # 强制保存所有数据
        await self.user_manager.force_save()
        logger.info("EmotionAI Pro 插件已安全关闭")