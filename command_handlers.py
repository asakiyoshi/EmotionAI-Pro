# command_handlers.py
import time
import asyncio
from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import asdict

from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import StarTools

from .models import EnhancedEmotionalState, RankingEntry
from .managers import UserStateManager, RankingManager, SmartUpdateManager, EmotionAnalyzer
from .memory import EnhancedMemorySystem
from .config import PluginConfig, PrivacyLevel

class BaseCommandHandler:
    """基础命令处理器"""
    
    def __init__(self, plugin):
        self.plugin = plugin
        self.config = plugin.config
        self.user_manager = plugin.user_manager

class UserCommandHandler(BaseCommandHandler):
    """用户命令处理器"""
    
    def __init__(self, plugin):
        super().__init__(plugin)
        self.ranking_manager = plugin.ranking_manager
        self.weight_manager = plugin.weight_manager
        self.analyzer = plugin.analyzer
    
    async def show_emotional_state(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """显示情感状态"""
        user_key = self.plugin._get_user_key(event)
        state = await self.user_manager.get_user_state(user_key)
    
        # 获取基础情感状态文本
        response_text = self.plugin._format_emotional_state(state)
    
        # 在回复开头添加用户标识
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
    
    async def toggle_status_display(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """切换状态显示开关"""
        user_key = self.plugin._get_user_key(event)
        state = await self.user_manager.get_user_state(user_key)
        state.show_status = not state.show_status
        await self.user_manager.update_user_state(user_key, state)
        
        status_text = "开启" if state.show_status else "关闭"
        yield event.plain_result(f"【状态显示】已{status_text}状态显示")
        event.stop_event()
    
    async def show_favor_ranking(self, event: AstrMessageEvent, num: str = "10") -> AsyncGenerator[Any, None]:
        """显示好感度排行榜"""
        try:
            limit = min(int(num), 20)
            if limit <= 0:
                raise ValueError
        except ValueError:
            yield event.plain_result("【错误】排行数量必须是一个正整数（最大20）。")
            event.stop_event()
            return

        rankings = await self.ranking_manager.get_enhanced_ranking(limit, True)
        
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
    
    async def show_negative_favor_ranking(self, event: AstrMessageEvent, num: str = "10") -> AsyncGenerator[Any, None]:
        """显示负好感排行榜"""
        try:
            limit = min(int(num), 20)
            if limit <= 0:
                raise ValueError
        except ValueError:
            yield event.plain_result("【错误】排行数量必须是一个正整数（最大20）。")
            event.stop_event()
            return

        rankings = await self.ranking_manager.get_enhanced_ranking(limit, False)
        
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
    
    async def show_relationship_stage(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """显示关系阶段详情"""
        user_key = self.plugin._get_user_key(event)
        state = await self.user_manager.get_user_state(user_key)
        
        # 这里需要原有的 weight_manager 逻辑，保持原有实现
        stage_info = self.weight_manager.get_stage_info(state)
        stage_advice = self.weight_manager.get_stage_progression_advice(state)
        
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
        ])
        
        yield event.plain_result("\n".join(response_lines))
        event.stop_event()

class AdminCommandHandler(BaseCommandHandler):
    """管理员命令处理器"""
    
    def __init__(self, plugin):
        super().__init__(plugin)
        self.ranking_manager = plugin.ranking_manager
        self.weight_manager = plugin.weight_manager
        self.analyzer = plugin.analyzer
    
    def _is_admin(self, event: AstrMessageEvent) -> bool:
        """检查管理员权限"""
        return event.role == "admin" or event.get_sender_id() in self.config.admin_qq_list
    
    def _resolve_user_key(self, user_input: str) -> str:
        """解析用户标识符"""
        return self.user_manager.resolve_user_key(user_input, self.config.session_based)
    
    async def set_favor(self, event: AstrMessageEvent, user_input: str, value: str) -> AsyncGenerator[Any, None]:
        """设置好感度"""
        if not self._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
            
        try:
            favor_value = int(value)
            if not self.config.favour_min <= favor_value <= self.config.favour_max:
                yield event.plain_result(f"【错误】好感度值必须在 {self.config.favour_min} 到 {self.config.favour_max} 之间。")
                event.stop_event()
                return
        except ValueError:
            yield event.plain_result("【错误】好感度值必须是数字")
            event.stop_event()
            return
            
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        
        state = await self.user_manager.get_user_state(user_key)
        state.favor = favor_value
        
        await self.user_manager.update_user_state(user_key, state)
        
        mode_info = "（会话模式）" if self.config.session_based else ""
        yield event.plain_result(f"【成功】用户 {user_input}{mode_info} 的好感度已设置为 {favor_value}")
        event.stop_event()
    
    async def set_intimacy(self, event: AstrMessageEvent, user_input: str, value: str) -> AsyncGenerator[Any, None]:
        """设置亲密度"""
        if not self._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
            
        try:
            intimacy_value = int(value)
            if not self.config.intimacy_min <= intimacy_value <= self.config.intimacy_max:
                yield event.plain_result(f"【错误】亲密度值必须在 {self.config.intimacy_min} 到 {self.config.intimacy_max} 之间。")
                event.stop_event()
                return
        except ValueError:
            yield event.plain_result("【错误】亲密度值必须是数字")
            event.stop_event()
            return
            
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        
        state = await self.user_manager.get_user_state(user_key)
        state.intimacy = intimacy_value
        
        await self.user_manager.update_user_state(user_key, state)
        
        mode_info = "（会话模式）" if self.config.session_based else ""
        yield event.plain_result(f"【成功】用户 {user_input}{mode_info} 的亲密度已设置为 {intimacy_value}")
        event.stop_event()
    
    async def set_attitude(self, event: AstrMessageEvent, user_input: str, attitude: str) -> AsyncGenerator[Any, None]:
        """设置态度"""
        if not self._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
            
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        
        state = await self.user_manager.get_user_state(user_key)
        state.descriptions.attitude = attitude
        
        await self.user_manager.update_user_state(user_key, state)
        
        mode_info = "（会话模式）" if self.config.session_based else ""
        yield event.plain_result(f"【成功】用户 {user_input}{mode_info} 的态度已设置为 {attitude}")
        event.stop_event()
    
    async def set_relationship(self, event: AstrMessageEvent, user_input: str, relationship: str) -> AsyncGenerator[Any, None]:
        """设置关系"""
        if not self._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
            
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        
        state = await self.user_manager.get_user_state(user_key)
        state.descriptions.relationship = relationship
        
        await self.user_manager.update_user_state(user_key, state)
        
        mode_info = "（会话模式）" if self.config.session_based else ""
        yield event.plain_result(f"【成功】用户 {user_input}{mode_info} 的关系已设置为 {relationship}")
        event.stop_event()
    
    async def set_global_privacy_level(self, event: AstrMessageEvent, level: str) -> AsyncGenerator[Any, None]:
        """设置全局隐私级别"""
        if not self._is_admin(event):
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
        self.config.global_privacy_level = PrivacyLevel(privacy_level)
        level_names = {0: "完全保密", 1: "基础显示", 2: "详细显示"}
        
        print(f"管理员更新全局隐私级别: {level_names[privacy_level]}")
        yield event.plain_result(f"【全局设置】隐私级别已设置为: {level_names[privacy_level]}（全员生效）")
        event.stop_event()
    
    async def reset_favor(self, event: AstrMessageEvent, user_input: str) -> AsyncGenerator[Any, None]:
        """重置用户好感度状态"""
        if not self._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
            
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        new_state = EnhancedEmotionalState(user_key=user_key)
        
        await self.user_manager.update_user_state(user_key, new_state)
        
        mode_info = "（会话模式）" if self.config.session_based else ""
        yield event.plain_result(f"【成功】用户 {user_input}{mode_info} 的情感状态已完全重置")
        event.stop_event()
    
    async def view_favor(self, event: AstrMessageEvent, user_input: str) -> AsyncGenerator[Any, None]:
        """管理员查看指定用户的好感状态"""
        if not self._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
        
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        state = await self.user_manager.get_user_state(user_key)
        
        # 使用动态权重计算情感档案
        stage_info = self.weight_manager.get_stage_info(state)
        profile = self.analyzer.get_emotional_profile(state, stage_info['favor_weight'], stage_info['intimacy_weight'])
        
        # 格式化显示名称
        display_name = self.ranking_manager._format_user_display(user_input)
        
        response_lines = [
            f"【用户 {display_name} 完整情感状态】",
            f"用户标识: {user_key}",
            "==================",
            f"关系阶段: {stage_info['stage_name']} (进度: {stage_info['progress_to_next']:.1f}%)",
            f"动态权重: 好感{stage_info['favor_weight']*100:.0f}% | 亲密{stage_info['intimacy_weight']*100:.0f}%",
            f"态度: {state.descriptions.attitude} | 关系: {state.descriptions.relationship}",
            f"好感度: {state.favor} | 亲密度: {state.intimacy}",
            f"复合评分: {profile['composite_score']:.1f}",
            f"主导情感: {profile['dominant_emotion']} | 情感强度: {profile['emotion_intensity']}%",
            f"互动统计: {state.stats.total_count}次 (正面: {state.stats.positive_count}, 负面: {state.stats.negative_count})",
            f"最后互动: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state.stats.last_interaction_time)) if state.stats.last_interaction_time > 0 else '从未互动'}",
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
            f"  喜悦: {state.emotions.joy} | 信任: {state.emotions.trust} | 恐惧: {state.emotions.fear} | 惊讶: {state.emotions.surprise}",
            f"  悲伤: {state.emotions.sadness} | 厌恶: {state.emotions.disgust} | 愤怒: {state.emotions.anger} | 期待: {state.emotions.anticipation}"
        ])
        
        yield event.plain_result("\n".join(response_lines))
        event.stop_event()
    
    async def reset_plugin(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """重置插件所有数据"""
        if not self._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        # 重置所有数据
        await self.user_manager.clear_all_data()
        
        print("管理员执行了插件重置操作")
        
        yield event.plain_result("【成功】插件所有数据已重置")
        event.stop_event()
    
    async def backup_data(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """备份插件数据"""
        if not self._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        try:
            backup_name = await self.plugin.backup_manager.create_backup()
            yield event.plain_result(f"【成功】数据备份成功: {backup_name}")
            event.stop_event()
        except Exception as e:
            yield event.plain_result(f"【错误】备份失败: {str(e)}")
            event.stop_event()

class DebugCommandHandler(BaseCommandHandler):
    """调试命令处理器"""
    
    def __init__(self, plugin):
        super().__init__(plugin)
        self.memory_system = plugin.memory_system
    
    async def show_cache_stats(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """显示缓存统计信息 - 增强版本"""
        try:
            stats = await self.user_manager.cache.get_stats()
            
            response = [
                "【缓存统计信息】",
                "==================",
                f"缓存条目: {stats.get('total_entries', 0)}",
                f"访问次数: {stats.get('access_count', 0)}",
                f"命中次数: {stats.get('hit_count', 0)}",
                f"命中率: {stats.get('hit_rate', 0):.2f}%",
                f"淘汰次数: {stats.get('eviction_count', 0)}",
                f"分片数量: {stats.get('shard_count', 0)}",
            ]
            
            # 添加内存使用信息
            memory_info = stats.get('memory_usage', {})
            if memory_info:
                response.extend([
                    "",
                    "【内存使用】",
                    f"总内存: {memory_info.get('total_bytes', 0) / 1024:.1f} KB",
                    f"最大内存: {memory_info.get('max_bytes', 0) / 1024:.1f} KB",
                    f"使用率: {memory_info.get('usage_percent', 0):.1f}%"
                ])
            
            # 添加分片分布
            distribution = stats.get('shard_stats', [])
            if distribution and len(distribution) > 0:
                response.extend([
                    "",
                    "【分片分布】",
                    f"平均条目: {stats.get('total_entries', 0) / len(distribution):.1f}",
                    f"最小条目: {min(s.get('total_entries', 0) for s in distribution)}",
                    f"最大条目: {max(s.get('total_entries', 0) for s in distribution)}"
                ])
            
            response.extend([
                "",
                "【管理器统计】",
                f"状态加载: {self.user_manager.stats.get('state_loads', 0)}",
                f"状态保存: {self.user_manager.stats.get('state_saves', 0)}",
                f"缓存命中: {self.user_manager.stats.get('cache_hits', 0)}",
                f"缓存未命中: {self.user_manager.stats.get('cache_misses', 0)}",
                f"错误次数: {self.user_manager.stats.get('errors', 0)}",
                "",
                f"提示: 缓存用于提高情感状态读取性能"
            ])
            
            yield event.plain_result("\n".join(response))
            
        except Exception as e:
            yield event.plain_result(f"【错误】获取缓存统计失败: {str(e)}")
        
        event.stop_event()
    
    async def debug_event(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """调试事件结构"""
        if not self._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
        
        debug_info = [
            "【事件调试信息】",
            "==================",
            f"事件类型: {type(event).__name__}",
            f"发送者ID: {event.get_sender_id()}",
            f"角色: {getattr(event, 'role', '未知')}",
        ]
    
        # 关键属性检查
        key_attrs = ['message_str', 'message_obj']
    
        debug_info.append("\n【关键属性检查】")
        for attr in key_attrs:
            if hasattr(event, attr):
                try:
                    value = getattr(event, attr)
                    debug_info.append(f"{attr}: {type(value)} = '{str(value)[:100]}'")
                except Exception as e:
                    debug_info.append(f"{attr}: 访问错误 - {e}")
            else:
                debug_info.append(f"{attr}: 不存在")
    
        # 测试消息提取
        debug_info.append("\n【消息提取测试】")
        test_result = self.plugin._get_message_text(event)
        debug_info.append(f"提取结果: '{test_result}'")
    
        yield event.plain_result("\n".join(debug_info))
        event.stop_event()
    
    async def debug_memory(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """调试记忆系统"""
        if not self._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
        
        user_key = self.plugin._get_user_key(event)
    
        debug_info = [
            "【记忆系统调试信息】",
            "==================",
            f"用户标识: {user_key}",
        ]
    
        try:
            # 测试短期记忆
            recent_context = await self.memory_system.get_recent_context(user_key)
            debug_info.append(f"\n近期上下文: {recent_context}")
        
            # 测试长期记忆
            long_term_context = self.memory_system.get_relationship_context(user_key)
            debug_info.append(f"\n长期关系上下文: {long_term_context}")
        
            # 获取记忆统计
            memory_stats = await self.memory_system.get_user_memory_stats(user_key)
            debug_info.append(f"\n记忆统计:")
            debug_info.append(f"  长期记忆: {memory_stats['long_term_count']}条")
            debug_info.append(f"  短期记忆: {memory_stats['recent_count']}条")
            debug_info.append(f"  重要事件: {memory_stats['important_count']}个")
            debug_info.append(f"  平均意义: {memory_stats['avg_significance']:.1f}")
        
        except Exception as e:
            debug_info.append(f"\n调试过程中出错: {e}")
    
        yield event.plain_result("\n".join(debug_info))
        event.stop_event()
    
    async def fix_interaction_stats(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """修复互动统计数据"""
        if not self._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
        
        user_key = self.plugin._get_user_key(event)
        state = await self.user_manager.get_user_state(user_key)
    
        # 基于当前好感度和亲密度估算正面互动
        if state.stats.total_count > 0:
            # 假设大部分互动都是正面的（因为好感度和亲密度在增长）
            estimated_positive = max(1, int(state.stats.total_count * 0.8))  # 80% 估算为正面
            state.stats.positive_count = min(estimated_positive, state.stats.total_count)
        
            # 计算正面互动比例
            positive_ratio = state.stats.positive_ratio
        
            await self.user_manager.update_user_state(user_key, state)
        
            yield event.plain_result(f"【成功】修复互动统计：正面互动 {state.stats.positive_count}/{state.stats.total_count} ({positive_ratio:.1f}%)")
        else:
            yield event.plain_result("【信息】暂无互动数据需要修复")
    
        event.stop_event()

    async def cleanup_initial_users(self, event: AstrMessageEvent):
        """手动清理初始状态用户缓存"""
        if not self._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
    
        try:
            yield event.plain_result("【清理中】正在扫描和清理初始状态用户缓存...")
        
            cleaned_count = await self.plugin.user_manager.smart_cache_cleanup()
        
            if cleaned_count > 0:
                yield event.plain_result(f"【成功】清理了 {cleaned_count} 个初始状态用户缓存")
            else:
                yield event.plain_result("【完成】没有找到需要清理的初始状态用户")
            
        except Exception as e:
            yield event.plain_result(f"【错误】清理失败: {e}")
    
        event.stop_event()    