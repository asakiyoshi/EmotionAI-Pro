# constants.py
"""常量定义"""
from enum import Enum

# 时间常量（秒）
class TimeConstants:
    ONE_MINUTE = 60
    FIVE_MINUTES = 300
    THIRTY_MINUTES = 1800
    ONE_HOUR = 3600
    ONE_DAY = 86400

# 缓存常量
class CacheConstants:
    DEFAULT_TTL = 300
    MAX_SIZE = 1000
    SHARD_COUNT = 8  # 缓存分片数

# 情感常量
class EmotionConstants:
    MIN_FAVOR = -100
    MAX_FAVOR = 100
    MIN_INTIMACY = 0
    MAX_INTIMACY = 100
    MIN_EMOTION = 0
    MAX_EMOTION = 100

# 更新阈值
class UpdateThresholds:
    MINOR_CHANGE = 3
    MAJOR_CHANGE = 8
    FORCE_UPDATE = 5
    EMOTIONAL_SIGNIFICANCE = 5

# 文件路径
class PathConstants:
    USER_DATA_FILE = "user_emotion_data.json"
    LONG_TERM_MEMORY_FILE = "long_term_memory.json"
    BACKUP_DIR = "backups"
    TEMP_DIR = "temp"