# __init__.py
"""
EmotionAI Pro 优化版插件
融合 EmotionAI 与 FavourPro 的高级情感智能交互系统
"""

import sys

# 检查Python版本
if sys.version_info < (3, 8):
    raise RuntimeError("EmotionAI Pro requires Python 3.8 or higher")

# 检查必要依赖
try:
    import pydantic
    if int(pydantic.__version__.split('.')[0]) < 2:
        raise ImportError("pydantic version 2.0.0 or higher is required")
except ImportError as e:
    raise ImportError(f"Missing required dependency: {e}")

try:
    import aiofiles
except ImportError:
    raise ImportError("Missing required dependency: aiofiles")

from .main import EmotionAIProPlugin

__version__ = "4.0.0"
__all__ = ['EmotionAIProPlugin', '__version__']