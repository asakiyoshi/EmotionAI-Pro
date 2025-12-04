# attitude_manager.py
from typing import Dict, Any

from .models import EnhancedEmotionalState

class AttitudeRelationshipManager:
    """态度关系管理器 - 完整的原有实现"""
    
    BEHAVIOR_STYLES = {
        "hostile": "极其简短、尖锐，可能拒绝回应",
        "cold": "冷淡、简洁、不耐烦", 
        "neutral": "客观、保持距离、标准化回应",
        "friendly": "积极、乐于协助、带有正面情绪",
        "intimate": "热情、主动、富有情感，可使用亲昵称呼"
    }
    
    TONE_INSTRUCTIONS = {
        "joy": "你的语气应该是愉快、充满热情和活力的。多使用积极的词汇和表情符号。",
        "trust": "你的语气应该是平和、真诚且令人安心的。展现出你的可靠和支持。",
        "fear": "你的语气应该显得有些紧张、谨慎或不安。表现出犹豫或退缩。",
        "surprise": "你的语气应该是震惊、难以置信或充满好奇的。多使用感叹号和反问。",
        "sadness": "你的语气应该是低落、消沉和遗憾的。句子要简短，带有一种无力感。",
        "disgust": "你的语气应该是厌烦、抗拒甚至带有生理性不适的。表现出想回避的态度。",
        "anger": "你的语气应该是愤怒、急躁和具有攻击性的。使用简短有力的句子，表现出不耐烦。",
        "anticipation": "你的语气应该是期待、急切和向往的。关注未来的可能性。"
    }
    
    @classmethod
    def get_behavior_guidance(cls, favor: int, intimacy: int) -> str:
        """根据好感度和亲密度获取行为指导"""
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
    def get_tone_instruction(cls, state: EnhancedEmotionalState) -> str:
        """根据主导情感获取语气指导"""
        emotions = {
            "joy": state.emotions.joy,
            "trust": state.emotions.trust,
            "fear": state.emotions.fear,
            "surprise": state.emotions.surprise,
            "sadness": state.emotions.sadness,
            "disgust": state.emotions.disgust,
            "anger": state.emotions.anger,
            "anticipation": state.emotions.anticipation
        }
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        
        if dominant_emotion[1] > 30:
            return cls.TONE_INSTRUCTIONS.get(dominant_emotion[0], "保持自然友好的语气。")
        else:
            if state.favor >= 40:
                return "保持友好积极的语气。"
            elif state.favor >= -10:
                return "保持中立客观的语气。"
            else:
                return "保持简洁冷淡的语气。"