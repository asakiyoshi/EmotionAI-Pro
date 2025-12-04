# emotion_expert.py
import re
import json
import random
import asyncio
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import asdict
import hashlib

from .cache import ShardedTTLCache
from .models import EnhancedEmotionalState
from .constants import TimeConstants

class EmotionAnalysisExpert:
    """情感分析专家 - 完全修复版本"""

    def __init__(self, cache: ShardedTTLCache, context=None, 
                 secondary_llm_provider: str = None, secondary_llm_model: str = None):
        self.cache = cache
        self.context = context
        self.secondary_llm_provider = secondary_llm_provider
        self.secondary_llm_model = secondary_llm_model
        self.llm_timeout = 30.0
        self.llm_retry_count = 3
        self.llm_retry_delay = 1.0
        self._llm_available = True  # 跟踪LLM可用性
        self._llm_failures = 0  # 连续失败次数

    async def analyze_and_update_emotion(self, user_key: str, user_message: str, ai_response: str,
                                       current_state: EnhancedEmotionalState) -> Dict[str, Any]:
        """情感分析入口 - 增强异常处理"""
        print(f"情感分析专家被调用: user_key={user_key}, message_length={len(user_message)}")
    
        # 生成更精确的缓存键，避免重复分析相同对话
        message_hash = hashlib.md5(f"{user_message}_{ai_response}".encode()).hexdigest()[:8]
        cache_key = f"emotion_analysis_{user_key}_{message_hash}"
        
        # 尝试从缓存获取
        cached = await self.cache.get(cache_key)
        if cached:
            print("使用缓存的情感分析结果")
            return cached
        
        # 主要分析流程
        analysis_result = None
        try:
            # 首先尝试使用真正的LLM分析
            if self._llm_available:
                analysis_result = await self._call_real_llm_with_retry(user_message, ai_response, current_state)
            
            # 如果LLM分析成功
            if analysis_result:
                updates = self._parse_emotion_analysis(analysis_result, current_state)
                updates['source'] = 'llm_analysis'
                updates['llm_available'] = True
                
                # 重置失败计数
                self._llm_failures = 0
                
            else:
                # LLM分析失败或不可用
                self._llm_failures += 1
                
                # 如果连续失败超过3次，暂时禁用LLM
                if self._llm_failures >= 3:
                    self._llm_available = False
                    print("LLM连续失败3次，暂时禁用LLM分析")
                
                # 使用智能后备方案
                updates = self._generate_smart_fallback(user_message, ai_response, current_state)
                updates['source'] = 'smart_fallback'
                updates['llm_available'] = False
                
        except Exception as e:
            print(f"情感分析过程发生异常: {e}")
            # 使用紧急后备方案
            updates = self._generate_emergency_fallback(user_message, ai_response, current_state)
            updates['source'] = 'emergency_fallback'
            updates['llm_available'] = False
        
        # 确保返回结果包含所有必要字段
        updates = self._ensure_updates_completeness(updates, current_state)
        
        # 缓存结果（短期缓存）
        await self.cache.set(cache_key, updates, ttl=TimeConstants.ONE_HOUR)
        print(f"情感分析完成: {updates.get('source', 'unknown')}")
        
        return updates

    async def _call_real_llm_with_retry(self, user_message: str, ai_response: str, 
                                      state: EnhancedEmotionalState) -> Optional[str]:
        """带重试的LLM调用"""
        for attempt in range(self.llm_retry_count):
            try:
                result = await self._call_real_llm(user_message, ai_response, state)
                if result and len(result) > 10:  # 确保有足够的返回内容
                    return result
                else:
                    print(f"LLM返回内容过短或为空: {result}")
            except asyncio.TimeoutError:
                print(f"LLM调用超时 (尝试 {attempt + 1}/{self.llm_retry_count})")
            except Exception as e:
                print(f"LLM调用异常 (尝试 {attempt + 1}/{self.llm_retry_count}): {e}")
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < self.llm_retry_count - 1:
                await asyncio.sleep(self.llm_retry_delay * (attempt + 1))
        
        print(f"经过 {self.llm_retry_count} 次尝试，LLM调用失败")
        return None

    async def _call_real_llm(self, user_message: str, ai_response: str, state: EnhancedEmotionalState) -> Optional[str]:
        """调用真正的LLM进行情感分析"""
        if not self.context:
            print("没有context，无法调用LLM")
            return None
        
        providers = self.context.get_all_providers()
        if not providers:
            print("没有可用的LLM提供商")
            return None
        
        # 确定目标提供商
        target_provider = self._find_target_provider(providers)
        if not target_provider:
            print("找不到目标LLM提供商")
            return None
        
        # 构建提示词
        prompt = self._build_emotion_analysis_prompt(user_message, ai_response, state)
        
        try:
            # 实际调用LLM
            result = await self._execute_llm_call(target_provider, prompt)
            return result
        except Exception as e:
            print(f"LLM调用执行失败: {e}")
            return None

    def _find_target_provider(self, providers: List) -> Optional[Any]:
        """查找目标LLM提供商"""
        # 优先使用配置的辅助LLM提供商
        if self.secondary_llm_provider:
            target_name = self.secondary_llm_provider.lower()
            for provider in providers:
                provider_name = self._get_provider_name(provider).lower()
                if target_name in provider_name:
                    print(f"找到配置的辅助LLM提供商: {self._get_provider_name(provider)}")
                    return provider
        
        # 其次查找deepseek_default
        for provider in providers:
            provider_name = self._get_provider_name(provider).lower()
            if 'deepseek' in provider_name or 'default' in provider_name:
                print(f"找到DeepSeek提供商: {self._get_provider_name(provider)}")
                return provider
        
        # 使用第一个可用的
        if providers:
            print(f"使用第一个可用提供商: {self._get_provider_name(providers[0])}")
            return providers[0]
        
        return None

    def _get_provider_name(self, provider) -> str:
        """获取提供商名称"""
        name = getattr(provider, 'name', '')
        if not name:
            name = getattr(provider, '__class__', '').__name__
        return name or '未知'

    async def _execute_llm_call(self, provider, prompt: str) -> Optional[str]:
        """执行LLM调用 - 完整实现"""
        provider_name = self._get_provider_name(provider)
        print(f"使用LLM提供商 [{provider_name}] 进行情感分析")
        
        # 记录提示词长度（用于调试）
        print(f"情感分析提示词长度: {len(prompt)} 字符")
        
        try:
            # 尝试调用text_chat方法（异步）
            if hasattr(provider, 'text_chat') and asyncio.iscoroutinefunction(provider.text_chat):
                print(f"调用 {provider_name}.text_chat()")
                result = await asyncio.wait_for(
                    provider.text_chat(prompt),
                    timeout=self.llm_timeout
                )
                text = self._extract_response_text(result)
                if text and len(text) > 10:
                    print(f"LLM情感分析成功，响应长度: {len(text)}")
                    return text
                else:
                    print("LLM返回空或过短的响应")
                    return None
            
            # 尝试调用chat_completion方法（异步）
            elif hasattr(provider, 'chat_completion') and asyncio.iscoroutinefunction(provider.chat_completion):
                print(f"调用 {provider_name}.chat_completion()")
                
                # 构建消息
                messages = [{"role": "user", "content": prompt}]
                result = await asyncio.wait_for(
                    provider.chat_completion(messages=messages),
                    timeout=self.llm_timeout
                )
                text = self._extract_response_text(result)
                if text and len(text) > 10:
                    print(f"LLM情感分析成功，响应长度: {len(text)}")
                    return text
            
            # 尝试同步方法
            elif hasattr(provider, 'text_chat') and not asyncio.iscoroutinefunction(provider.text_chat):
                print(f"调用同步方法 {provider_name}.text_chat()")
                result = provider.text_chat(prompt)
                text = self._extract_response_text(result)
                if text and len(text) > 10:
                    return text
            
            print(f"提供商 {provider_name} 不支持已知的调用方法")
            return None
            
        except asyncio.TimeoutError:
            print(f"LLM调用超时 ({self.llm_timeout}秒)")
            return None
        except Exception as e:
            print(f"LLM调用异常: {e}")
            return None

    def _extract_response_text(self, response_obj) -> str:
        """从LLM响应对象中提取文本"""
        if isinstance(response_obj, str):
            return response_obj.strip()
        
        # 尝试常见属性
        text_attrs = ['completion_text', 'text', 'content', 'response', 'result', 'message', 'choices']
        
        for attr in text_attrs:
            if hasattr(response_obj, attr):
                val = getattr(response_obj, attr)
                if isinstance(val, str):
                    return val.strip()
                elif isinstance(val, list) and len(val) > 0:
                    # 处理choices数组
                    first_choice = val[0]
                    if hasattr(first_choice, 'message'):
                        msg = first_choice.message
                        if hasattr(msg, 'content'):
                            return msg.content.strip()
        
        # 尝试字典访问
        if isinstance(response_obj, dict):
            for attr in text_attrs:
                if attr in response_obj:
                    val = response_obj[attr]
                    if isinstance(val, str):
                        return val.strip()
                    elif isinstance(val, list) and len(val) > 0:
                        if 'message' in val[0] and 'content' in val[0]['message']:
                            return val[0]['message']['content'].strip()
        
        # 最后尝试字符串转换
        try:
            text = str(response_obj).strip()
            if text and len(text) > 10:
                return text
        except:
            pass
        
        return ""

    def _generate_smart_fallback(self, user_message: str, ai_response: str, state: EnhancedEmotionalState) -> Dict[str, Any]:
        """智能后备方案 - 返回字典而非字符串"""
        user_lower = user_message.lower()
        resp_lower = ai_response.lower()
        
        # 情感关键词分析
        positive_words = ['好', '开心', '高兴', '谢谢', '感谢', '喜欢', '爱', '不错', '棒', '可爱', '漂亮', '美丽', '相信']
        negative_words = ['讨厌', '生气', '愤怒', '烦', '恨', '滚', '傻', '笨', '蠢', '垃圾', '不愿意']
        intimate_words = ['想你', '想念', '关心', '担心', '在乎', '重要', '宝贝', '亲爱的', '搞好关系']
        
        # 计算情感权重
        pos_weight = sum(3 for word in positive_words if word in user_lower) + \
                    sum(1 for word in positive_words if word in resp_lower)
        neg_weight = sum(3 for word in negative_words if word in user_lower) + \
                    sum(1 for word in negative_words if word in resp_lower)
        int_weight = sum(2 for word in intimate_words if word in user_lower) + \
                    sum(1 for word in intimate_words if word in resp_lower)
        
        # 基于权重生成响应
        emotion_updates = {}
        
        if neg_weight > pos_weight and neg_weight > 0:
            # 负面互动
            neg_strength = min(3, neg_weight)
            emotion_updates = {
                "favor": -neg_strength,
                "intimacy": -1,
                "sadness": 2,
                "anger": 1,
                "disgust": 1
            }
            relationship = "关系紧张"
            attitude = "谨慎回应"
            
        elif pos_weight > neg_weight and pos_weight > 0:
            # 正面互动
            pos_strength = min(3, pos_weight)
            intimacy_boost = 2 if int_weight > 0 else 1
            emotion_updates = {
                "favor": pos_strength,
                "intimacy": intimacy_boost,
                "joy": 2,
                "trust": 1,
                "anticipation": 1
            }
            relationship = "友好的对话伙伴"
            attitude = "愉快开放的交流"
            
        elif int_weight > 0:
            # 亲密互动
            emotion_updates = {
                "favor": 1,
                "intimacy": 3,
                "joy": 2,
                "trust": 2,
                "anticipation": 1
            }
            relationship = "亲密的朋友"
            attitude = "温暖关怀的交流"
            
        else:
            # 中性互动 - 小幅正面
            emotion_updates = {
                "favor": 0,
                "intimacy": 0,
                "anticipation": 1
            }
            relationship = "平常的交流对象"
            attitude = "标准回应"
        
        return {
            **emotion_updates,
            "relationship_text": relationship,
            "attitude_text": attitude
        }

    def _generate_emergency_fallback(self, user_message: str, ai_response: str, state: EnhancedEmotionalState) -> Dict[str, Any]:
        """紧急后备方案 - 只在完全失败时使用"""
        # 默认产生小幅正面情感变化
        return {
            'favor': 1,
            'intimacy': 1,
            'joy': 1,
            'trust': 0,
            'fear': 0,
            'surprise': 0,
            'sadness': 0,
            'disgust': 0,
            'anger': 0,
            'anticipation': 1,
            'relationship_text': "正常关系",
            'attitude_text': "友好交流"
        }

    def _build_emotion_analysis_prompt(self, user_msg: str, bot_msg: str, state: EnhancedEmotionalState) -> str:
        """构建生动的情感分析提示词"""
        return f"""你是一个情感分析专家，请分析以下对话的情感变化，输出JSON格式的分析结果。

对话内容：
用户：「{user_msg}」
AI：「{bot_msg}」

当前用户情感状态：
- 好感度：{state.favor}（范围：-100到100）
- 亲密度：{state.intimacy}（范围：0到100）
- 互动次数：{state.stats.total_count}次
- 正面互动比例：{state.stats.positive_ratio:.1f}%

【情感数值变化范围】
请为以下情感维度分配-2到+2之间的整数值：
- 好感度 (favor): 基于对话的情感倾向
- 亲密度 (intimacy): 基于关系的亲密程度
- 喜悦 (joy): 愉快、开心的程度
- 信任 (trust): 信任、可靠的程度
- 恐惧 (fear): 害怕、担忧的程度
- 惊讶 (surprise): 惊讶、意外的程度
- 悲伤 (sadness): 伤心、难过的程度
- 厌恶 (disgust): 厌恶、反感的程度
- 愤怒 (anger): 生气、愤怒的程度
- 期待 (anticipation): 期待、盼望的程度

【关系描述要求】
- 描述双方的关系性质，保持生动有趣
- 考虑当前好感度、亲密度和互动历史
- 避免过于复杂冗长的描述
- 保持自然、符合人类社交常识

【态度描述要求】
- 描述AI对用户的回应态度或互动方式
- 体现情感倾向和互动风格
- 保持简洁明了

【输出格式】
请输出严格的JSON格式：
{{
  "emotion_updates": {{
    "favor": 整数变化值,
    "intimacy": 整数变化值,
    "joy": 整数变化值,
    "trust": 整数变化值,
    "fear": 整数变化值,
    "surprise": 整数变化值,
    "sadness": 整数变化值,
    "disgust": 整数变化值,
    "anger": 整数变化值,
    "anticipation": 整数变化值
  }},
  "relationship": "关系描述",
  "attitude": "态度描述"
}}

注意：如果对话情感不明显，可以设置部分值为0。"""

    def _parse_emotion_analysis(self, analysis_text: str, current_state: EnhancedEmotionalState) -> Dict[str, Any]:
        """解析情感分析结果 - 增强版本"""
        updates = {}
        
        try:
            # 清理和提取JSON
            cleaned_text = self._clean_json_response(analysis_text)
            
            # 尝试解析JSON
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                json_str = self._fix_common_json_errors(json_str)
                
                data = json.loads(json_str)
                
                # 验证必需字段
                if 'emotion_updates' not in data:
                    raise ValueError("缺少emotion_updates字段")
                
                # 解析数值更新
                emotion_updates = data['emotion_updates']
                valid_emotions = ['favor', 'intimacy', 'joy', 'trust', 'fear', 
                                 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
                
                for emotion in valid_emotions:
                    if emotion in emotion_updates:
                        try:
                            value = emotion_updates[emotion]
                            # 确保是整数且在合理范围内
                            if isinstance(value, (int, float)):
                                int_value = int(value)
                                # 限制变化范围
                                if emotion in ['favor', 'intimacy']:
                                    int_value = max(-5, min(5, int_value))
                                else:
                                    int_value = max(-3, min(3, int_value))
                                updates[emotion] = int_value
                        except (ValueError, TypeError):
                            updates[emotion] = 0
                    else:
                        updates[emotion] = 0  # 缺失的情感设为0
                
                # 解析文本描述
                if 'relationship' in data and data['relationship']:
                    updates['relationship_text'] = str(data['relationship']).strip()[:50]  # 限制长度
                else:
                    updates['relationship_text'] = "正常关系"
                
                if 'attitude' in data and data['attitude']:
                    updates['attitude_text'] = str(data['attitude']).strip()[:30]  # 限制长度
                else:
                    updates['attitude_text'] = "友好交流"
                
                print(f"成功解析JSON情感分析结果，包含 {len(updates)} 个更新")
                return updates
                
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}, 文本: {analysis_text[:100]}")
        except KeyError as e:
            print(f"JSON字段缺失: {e}")
        except Exception as e:
            print(f"解析情感分析结果失败: {e}")
        
        # JSON解析失败，尝试从文本提取
        print("JSON解析失败，尝试文本提取")
        return self._extract_updates_from_text(analysis_text)

    def _clean_json_response(self, text: str) -> str:
        """清理JSON响应"""
        # 移除可能的代码块标记
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```\s*', '', text)
        
        # 移除JSON之前的说明文字
        text = re.sub(r'^[^{]*', '', text)
        text = re.sub(r'[^}]*$', '', text)
        
        return text.strip()

    def _fix_common_json_errors(self, json_str: str) -> str:
        """修复常见的JSON格式错误"""
        # 修复未加引号的键
        json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
        
        # 修复单引号
        json_str = json_str.replace("'", '"')
        
        # 修复多余的逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 确保大括号匹配
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        
        return json_str

    def _extract_updates_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取更新信息（后备方法）"""
        updates = {
            'favor': 0,
            'intimacy': 0,
            'joy': 0,
            'trust': 0,
            'fear': 0,
            'surprise': 0,
            'sadness': 0,
            'disgust': 0,
            'anger': 0,
            'anticipation': 0,
            'relationship_text': "正常关系",
            'attitude_text': "友好交流"
        }
        
        text_lower = text.lower()
        
        # 简单的情感关键词映射
        emotion_patterns = [
            ('favor', ['好感', '喜欢', '欣赏', '满意', 'positive']),
            ('intimacy', ['亲密', '亲近', '密切', '亲密感', 'intimacy']),
            ('joy', ['开心', '高兴', '愉快', '欢乐', '微笑', 'joy', 'happy']),
            ('trust', ['信任', '相信', '可靠', '安心', 'trust']),
            ('fear', ['害怕', '恐惧', '担心', '紧张', 'fear', 'scared']),
            ('surprise', ['惊讶', '惊奇', '意外', '吃惊', 'surprise']),
            ('sadness', ['悲伤', '伤心', '难过', '沮丧', 'sadness', 'sad']),
            ('disgust', ['厌恶', '讨厌', '反感', '恶心', 'disgust']),
            ('anger', ['生气', '愤怒', '恼火', '气愤', 'anger', 'angry']),
            ('anticipation', ['期待', '期望', '盼望', 'anticipation'])
        ]
        
        for emotion, keywords in emotion_patterns:
            for keyword in keywords:
                if keyword in text_lower:
                    # 根据关键词强度调整值
                    if '非常' in text_lower or '特别' in text_lower or 'extremely' in text_lower:
                        updates[emotion] = 2 if emotion in ['favor', 'intimacy'] else 1
                    elif '有点' in text_lower or '稍微' in text_lower or 'slightly' in text_lower:
                        updates[emotion] = 1 if emotion in ['favor', 'intimacy'] else 0
                    else:
                        updates[emotion] = 1 if emotion in ['favor', 'intimacy'] else 0
                    break
        
        return updates

    def _ensure_updates_completeness(self, updates: Dict[str, Any], state: EnhancedEmotionalState) -> Dict[str, Any]:
        """确保更新结果包含所有必要字段"""
        required_fields = {
            'favor': 0,
            'intimacy': 0,
            'joy': 0,
            'trust': 0,
            'fear': 0,
            'surprise': 0,
            'sadness': 0,
            'disgust': 0,
            'anger': 0,
            'anticipation': 0
        }
        
        # 确保所有情感字段都存在
        for field, default_value in required_fields.items():
            if field not in updates:
                updates[field] = default_value
        
        # 确保有文本描述
        if 'relationship_text' not in updates:
            updates['relationship_text'] = state.descriptions.relationship
        
        if 'attitude_text' not in updates:
            updates['attitude_text'] = state.descriptions.attitude
        
        # 确保有来源信息
        if 'source' not in updates:
            updates['source'] = 'unknown'
        
        if 'llm_available' not in updates:
            updates['llm_available'] = False
        
        return updates
    
    def reset_llm_availability(self):
        """重置LLM可用性状态"""
        self._llm_available = True
        self._llm_failures = 0
        print("已重置LLM可用性状态")