# migration.py
import json
import asyncio
from pathlib import Path
from typing import Dict, Any

async def migrate_old_data(old_data_path: Path, new_repository):
    """迁移旧数据到新格式"""
    old_user_file = old_data_path / "user_emotion_data.json"
    
    if not old_user_file.exists():
        print("没有找到旧数据文件")
        return
    
    try:
        with open(old_user_file, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
        
        migrated_count = 0
        for user_key, old_state in old_data.items():
            if isinstance(old_state, dict):
                # 转换为新格式
                new_state = _convert_to_new_format(user_key, old_state)
                if new_state:
                    await new_repository.save_user_state(user_key, new_state)
                    migrated_count += 1
        
        print(f"成功迁移 {migrated_count} 个用户数据")
        
    except Exception as e:
        print(f"数据迁移失败: {e}")

def _convert_to_new_format(user_key: str, old_data: Dict[str, Any]):
    """将旧数据格式转换为新格式"""
    from .models import EnhancedEmotionalState, EmotionalMetrics, InteractionStats, TextDescriptions
    
    try:
        # 基础字段
        favor = old_data.get('favor', 0)
        intimacy = old_data.get('intimacy', 0)
        
        # 情感指标
        emotions = EmotionalMetrics(
            joy=old_data.get('joy', 0),
            trust=old_data.get('trust', 0),
            fear=old_data.get('fear', 0),
            surprise=old_data.get('surprise', 0),
            sadness=old_data.get('sadness', 0),
            disgust=old_data.get('disgust', 0),
            anger=old_data.get('anger', 0),
            anticipation=old_data.get('anticipation', 0)
        )
        
        # 互动统计
        stats = InteractionStats(
            total_count=old_data.get('interaction_count', 0),
            positive_count=old_data.get('positive_interactions', 0),
            negative_count=old_data.get('negative_interactions', 0),
            last_interaction_time=old_data.get('last_interaction', 0)
        )
        
        # 文本描述
        descriptions = TextDescriptions(
            attitude=old_data.get('attitude', '中立'),
            relationship=old_data.get('relationship', '陌生人'),
            last_attitude_update=old_data.get('last_attitude_update', 0),
            last_relationship_update=old_data.get('last_relationship_update', 0),
            update_count=old_data.get('attitude_update_count', 0) + old_data.get('relationship_update_count', 0)
        )
        
        return EnhancedEmotionalState(
            user_key=user_key,
            favor=favor,
            intimacy=intimacy,
            emotions=emotions,
            stats=stats,
            descriptions=descriptions,
            relationship_stage=old_data.get('relationship_stage', '初识期'),
            stage_composite_score=old_data.get('stage_composite_score', 0.0),
            stage_progress=old_data.get('stage_progress', 0.0),
            force_update_counter=old_data.get('force_update_counter', 0),
            last_force_update=old_data.get('last_force_update', 0),
            show_status=old_data.get('show_status', False),
            privacy_level=old_data.get('privacy_level'),
            _previous_stage=old_data.get('_previous_stage'),
            _previous_composite=old_data.get('_previous_composite', 0.0)
        )
    
    except Exception as e:
        print(f"转换用户 {user_key} 数据失败: {e}")
        return None