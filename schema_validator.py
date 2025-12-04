# schema_validator.py
"""
配置文件验证器
"""
import json
from pathlib import Path
from typing import Dict, Any
from jsonschema import validate, ValidationError
import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    """配置文件验证器"""
    
    # 配置架构定义
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "session_based": {"type": "boolean"},
            "favour_min": {"type": "integer", "minimum": -1000, "maximum": 0},
            "favour_max": {"type": "integer", "minimum": 0, "maximum": 1000},
            "intimacy_min": {"type": "integer", "minimum": 0, "maximum": 100},
            "intimacy_max": {"type": "integer", "minimum": 0, "maximum": 1000},
            "change_min": {"type": "integer", "minimum": -100, "maximum": 0},
            "change_max": {"type": "integer", "minimum": 0, "maximum": 100},
            "admin_qq_list": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[0-9]{5,12}$"}
            },
            "plugin_priority": {"type": "integer", "minimum": 1, "maximum": 1000000},
            "enable_attitude_system": {"type": "boolean"},
            "enable_ai_text_generation": {"type": "boolean"},
            "global_privacy_level": {"type": "integer", "minimum": 0, "maximum": 2},
            "enable_smart_update": {"type": "boolean"},
            "force_update_interval": {"type": "integer", "minimum": 1, "maximum": 100},
            "emotional_significance_threshold": {"type": "integer", "minimum": 1, "maximum": 10},
            "enable_secondary_llm": {"type": "boolean"},
            "secondary_llm_provider": {"type": "string"},
            "secondary_llm_model": {"type": "string"},
            "cache_ttl": {"type": "integer", "minimum": 1},
            "cache_max_size": {"type": "integer", "minimum": 10},
            "auto_save_interval": {"type": "integer", "minimum": 1},
            "max_dirty_keys": {"type": "integer", "minimum": 10},
            "backup_retention_days": {"type": "integer", "minimum": 1}
        },
        "required": [
            "session_based", "favour_min", "favour_max", "intimacy_min", "intimacy_max",
            "change_min", "change_max", "admin_qq_list", "plugin_priority"
        ],
        "additionalProperties": False
    }
    
    @classmethod
    def validate_config_file(cls, config_path: Path) -> bool:
        """验证配置文件"""
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            return False
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 验证架构
            validate(instance=config_data, schema=cls.CONFIG_SCHEMA)
            
            # 额外验证：确保最小值小于最大值
            if config_data['favour_min'] >= config_data['favour_max']:
                logger.error(f"favour_min ({config_data['favour_min']}) 必须小于 favour_max ({config_data['favour_max']})")
                return False
            
            if config_data['intimacy_min'] >= config_data['intimacy_max']:
                logger.error(f"intimacy_min ({config_data['intimacy_min']}) 必须小于 intimacy_max ({config_data['intimacy_max']})")
                return False
            
            if config_data['change_min'] >= config_data['change_max']:
                logger.error(f"change_min ({config_data['change_min']}) 必须小于 change_max ({config_data['change_max']})")
                return False
            
            logger.info("配置文件验证成功")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"配置文件JSON格式错误: {e}")
            return False
        except ValidationError as e:
            logger.error(f"配置文件验证失败: {e.message}")
            return False
        except Exception as e:
            logger.error(f"配置文件验证过程中出错: {e}")
            return False
    
    @classmethod
    def create_default_config(cls, config_path: Path):
        """创建默认配置文件"""
        default_config = {
            "session_based": False,
            "favour_min": -100,
            "favour_max": 100,
            "intimacy_min": 0,
            "intimacy_max": 100,
            "change_min": -10,
            "change_max": 5,
            "admin_qq_list": [],
            "plugin_priority": 100000,
            "enable_attitude_system": True,
            "enable_ai_text_generation": True,
            "global_privacy_level": 1,
            "enable_smart_update": True,
            "force_update_interval": 5,
            "emotional_significance_threshold": 5,
            "enable_secondary_llm": True,
            "secondary_llm_provider": "",
            "secondary_llm_model": "",
            "cache_ttl": 300,
            "cache_max_size": 1000,
            "auto_save_interval": 60,
            "max_dirty_keys": 1000,
            "backup_retention_days": 7
        }
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"已创建默认配置文件: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建默认配置文件失败: {e}")
            return False