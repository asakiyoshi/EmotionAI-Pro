# examples/usage_example.py
"""
EmotionAI Pro 使用示例
"""

import asyncio
from emotionai_pro import EmotionAIProPlugin
from astrbot.api import AstrBotConfig

async def main():
    # 创建配置
    config = AstrBotConfig(
        session_based=False,
        favour_min=-100,
        favour_max=100,
        admin_qq_list=["123456789"]
    )
    
    # 创建插件实例
    plugin = EmotionAIProPlugin(context=None, config=config)
    
    try:
        # 模拟用户交互
        from astrbot.api.event import AstrMessageEvent
        from unittest.mock import Mock
        
        # 创建模拟事件
        event = Mock(spec=AstrMessageEvent)
        event.get_sender_id.return_value = "user123"
        event.message_str = "你好，今天心情怎么样？"
        event.role = "user"
        
        # 处理消息
        async for result in plugin.show_emotional_state(event):
            print(f"响应: {result}")
            
    finally:
        # 清理资源
        await plugin.terminate()

if __name__ == "__main__":
    asyncio.run(main())