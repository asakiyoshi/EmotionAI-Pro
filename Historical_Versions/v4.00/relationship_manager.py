# relationship_manager.py
import time
from typing import Dict, Any, Optional, Tuple

from .models import EnhancedEmotionalState

class DynamicWeightManager:
    """动态权重管理器 - 完整的原有实现"""
    
    # 关系阶段配置
    STAGE_CONFIGS = {
        "INITIAL": {
            "name": "初识期",
            "description": "好感驱动，建立吸引",
            "favor_weight": 0.7,
            "intimacy_weight": 0.3,
            "favor_range": (0, 40),
            "intimacy_range": (0, 30),
            "composite_threshold": 25,
            "transition_buffer": 3,
            "intimacy_boost_factor": 4.0
        },
        "DEEPENING": {
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
        "COMMITMENT": {
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
        "SYMBIOSIS": {
            "name": "共生期",
            "description": "完全融合，不分彼此",
            "favor_weight": 0.5,
            "intimacy_weight": 0.5,
            "favor_range": (90, 100),
            "intimacy_range": (85, 100),
            "composite_threshold": 95,
            "transition_buffer": 10,
            "intimacy_boost_factor": 1.0
        }
    }
    
    @classmethod
    def calculate_stage(cls, state: EnhancedEmotionalState) -> Tuple[str, Dict[str, Any]]:
        """计算当前关系阶段和过渡状态 - 保持原有逻辑"""
        current_composite = cls._calculate_raw_composite(state)
        previous_stage = state._previous_stage or "INITIAL"
        previous_composite = state._previous_composite
        
        # 判断当前阶段
        target_stage = cls._get_stage_by_score(current_composite, state)
        
        # 检查是否处于阶段过渡期
        transition_info = cls._check_transition_status(
            state, previous_stage, target_stage, previous_composite, current_composite
        )
        
        return target_stage, transition_info
    
    @classmethod
    def _calculate_raw_composite(cls, state: EnhancedEmotionalState) -> float:
        """计算原始复合评分"""
        current_stage = cls._get_stage_by_score(
            state.favor * 0.6 + state.intimacy * 0.4,
            state
        )
        stage_config = cls.STAGE_CONFIGS[current_stage]
        return state.favor * stage_config["favor_weight"] + state.intimacy * stage_config["intimacy_weight"]
    
    @classmethod
    def _get_stage_by_score(cls, composite_score: float, state: EnhancedEmotionalState) -> str:
        """滞后版阶段判定：上升阈值 > 下降阈值，防止抖动"""
        prev_stage = state._previous_stage or "INITIAL"

        # 计算当前"裸"阶段
        if composite_score >= cls.STAGE_CONFIGS["SYMBIOSIS"]["composite_threshold"]:
            raw_target = "SYMBIOSIS"
        elif composite_score >= cls.STAGE_CONFIGS["COMMITMENT"]["composite_threshold"]:
            raw_target = "COMMITMENT"
        elif composite_score >= cls.STAGE_CONFIGS["DEEPENING"]["composite_threshold"]:
            raw_target = "DEEPENING"
        else:
            raw_target = "INITIAL"

        # 滞后逻辑
        stage_values = {"INITIAL": 0, "DEEPENING": 1, "COMMITMENT": 2, "SYMBIOSIS": 3}
        UP_THRESHOLD = cls.STAGE_CONFIGS[raw_target]["composite_threshold"]
        DOWN_THRESHOLD = UP_THRESHOLD - 5  # 5 点滞后带
        
        # 如果比上一阶段高，用上升阈值；否则用下降阈值
        use_threshold = UP_THRESHOLD if stage_values[raw_target] > stage_values[prev_stage] else DOWN_THRESHOLD

        if composite_score < use_threshold:
            return prev_stage

        return raw_target
    
    @classmethod
    def _check_transition_status(cls, state: EnhancedEmotionalState, previous_stage: str, 
                               target_stage: str, previous_composite: float, 
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
            
        return transition_info
    
    @classmethod
    def _calculate_needed_intimacy(cls, state: EnhancedEmotionalState, target_config: Dict[str, Any], 
                                 protected_score: float) -> int:
        """计算达到目标阶段所需的最小亲密度"""
        fav_weight = target_config["favor_weight"]
        int_weight = target_config["intimacy_weight"]
        
        if int_weight == 0:
            return 0
            
        needed_intimacy = (protected_score - state.favor * fav_weight) / int_weight
        needed_intimacy = max(0, needed_intimacy)
        needed_intimacy = min(100, needed_intimacy)
        
        needed_intimacy = int(needed_intimacy)
        current_intimacy = state.intimacy
        
        return max(0, needed_intimacy - current_intimacy)
    
    @classmethod
    def _calculate_transition_progress(cls, state: EnhancedEmotionalState, target_config: Dict[str, Any],
                                    needed_intimacy: int) -> float:
        """计算过渡进度"""
        if needed_intimacy <= 0:
            return 100.0
            
        target_intimacy = state.intimacy + needed_intimacy
        current_progress = (state.intimacy / target_intimacy) * 100 if target_intimacy > 0 else 0
        return min(100.0, current_progress)
    
    @classmethod
    def get_stage_weights(cls, state: EnhancedEmotionalState) -> Tuple[float, float]:
        """获取当前阶段的权重（考虑过渡期）"""
        if state.favor < 0:
            return 1.0, 0.0
        
        target_stage, transition_info = cls.calculate_stage(state)
        stage_config = cls.STAGE_CONFIGS[target_stage]
        
        if transition_info["intimacy_boost_active"]:
            boost_factor = stage_config["intimacy_boost_factor"]
            base_favor = stage_config["favor_weight"]
            base_intimacy = stage_config["intimacy_weight"]
            
            total = base_favor + base_intimacy * boost_factor
            adjusted_favor = base_favor / total
            adjusted_intimacy = (base_intimacy * boost_factor) / total
            
            return adjusted_favor, adjusted_intimacy
        
        return stage_config["favor_weight"], stage_config["intimacy_weight"]
    
    @classmethod
    def calculate_composite_score(cls, state: EnhancedEmotionalState) -> float:
        """计算当前阶段的复合评分（应用过渡保护）"""
        if state.favor < 0:
            return state.favor
        
        target_stage, transition_info = cls.calculate_stage(state)
        return transition_info["protected_composite"]
    
    @classmethod
    def get_stage_info(cls, state: EnhancedEmotionalState) -> Dict[str, Any]:
        """获取完整的阶段信息（包含过渡状态）"""
        if state.favor < 0:
            return cls._get_negative_favor_stage_info(state)
        
        target_stage, transition_info = cls.calculate_stage(state)
        stage_config = cls.STAGE_CONFIGS[target_stage]
        
        favor_weight, intimacy_weight = cls.get_stage_weights(state)
        composite_score = cls.calculate_composite_score(state)
        
        progress = (composite_score / stage_config["composite_threshold"]) * 100
        progress_to_next = max(0, min(100, progress))
        
        info = {
            "stage": target_stage,
            "stage_name": stage_config["name"],
            "description": stage_config["description"],
            "favor_weight": favor_weight,
            "intimacy_weight": intimacy_weight,
            "composite_score": composite_score,
            "next_stage_threshold": stage_config["composite_threshold"],
            "progress_to_next": progress_to_next,
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
        composite_score = state.favor
        
        if state.favor >= -30:
            stage_name = "冷淡期"
            description = "关系冷淡，需要修复"
            progress = max(0, (state.favor + 30) / 30 * 100)
        elif state.favor >= -70:
            stage_name = "反感期"
            description = "存在反感情绪"
            progress = max(0, (state.favor + 70) / 40 * 100)
        else:
            stage_name = "敌对期"
            description = "关系敌对"
            progress = 0
        
        return {
            "stage": None,
            "stage_name": stage_name,
            "description": description,
            "favor_weight": 1.0,
            "intimacy_weight": 0.0,
            "composite_score": composite_score,
            "next_stage_threshold": 0,
            "progress_to_next": progress,
            "is_transitioning": False,
            "transition_progress": 0.0,
            "intimacy_boost_active": False,
            "needed_intimacy_boost": 0
        }
    
    @classmethod
    def apply_transition_benefits(cls, state: EnhancedEmotionalState, updates: Dict[str, Any]) -> Dict[str, Any]:
        """应用过渡期增益效果"""
        target_stage, transition_info = cls.calculate_stage(state)
        
        if transition_info["intimacy_boost_active"]:
            stage_config = cls.STAGE_CONFIGS[target_stage]
            boost_factor = stage_config["intimacy_boost_factor"]
            
            if 'intimacy' in updates:
                original_boost = updates['intimacy']
                boosted_boost = int(original_boost * boost_factor)
                updates['intimacy'] = boosted_boost
            
            if ('joy' in updates or 'trust' in updates or 'anticipation' in updates) and 'intimacy' not in updates:
                auto_intimacy = max(1, int(2 * boost_factor))
                updates['intimacy'] = updates.get('intimacy', 0) + auto_intimacy
        
        return updates
    
    @classmethod
    def get_stage_progression_advice(cls, state: EnhancedEmotionalState) -> str:
        """获取阶段进阶建议（包含过渡期建议）"""
        stage_info = cls.get_stage_info(state)
    
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
    
        advice_map = {
            "INITIAL": 
                "初识期：多展示个人魅力，建立良好第一印象。通过有趣的话题和积极的互动提升好感度。",
            "DEEPENING":
                "深化期：分享更多个人经历和情感，建立信任基础。共同经历和深度交流是关键。", 
            "COMMITMENT":
                "承诺期：巩固信任和默契，在困难时刻相互支持。关系的深度比广度更重要。",
            "SYMBIOSIS":
                "共生期：维持情感的深度连接，共同成长和创造美好回忆。"
        }
    
        return advice_map.get(stage_info["stage"], "继续培养这段关系吧！")