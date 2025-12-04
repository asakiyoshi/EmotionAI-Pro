# config_manager.py
import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import hashlib

from .config import PluginConfig

class ConfigManager:
    """配置管理器 - 增强的热重载支持"""
    
    def __init__(self, config_path: Path, initial_config: PluginConfig):
        self.config_path = config_path
        self.current_config = initial_config
        self._listeners: List[Callable] = []
        self._watch_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._last_hash: Optional[str] = None
        self._last_modified: float = 0
        self._error_count = 0
        self._last_error_time = 0
        
        # 确保配置文件存在
        self._ensure_config_file()
        
        print(f"配置管理器初始化完成，配置文件: {config_path}")
    
    def _ensure_config_file(self):
        """确保配置文件存在"""
        if not self.config_path.exists():
            print(f"配置文件不存在，创建默认配置: {self.config_path}")
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_config(self.current_config)
    
    def _calculate_config_hash(self, config_data: Dict[str, Any]) -> str:
        """计算配置哈希值"""
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _save_config(self, config: PluginConfig):
        """保存配置到文件"""
        try:
            config_dict = config.dict()
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
            
            # 更新哈希值
            self._last_hash = self._calculate_config_hash(config_dict)
            self._last_modified = time.time()
            
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def add_change_listener(self, callback: Callable):
        """添加配置变更监听器"""
        if callback not in self._listeners:
            self._listeners.append(callback)
            print(f"添加配置变更监听器: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def remove_change_listener(self, callback: Callable):
        """移除配置变更监听器"""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    async def start_watching(self, interval: float = 2.0):
        """开始监控配置文件变化"""
        if self._watch_task:
            print("配置监控任务已在运行")
            return
        
        print(f"开始监控配置文件变化，检查间隔: {interval}秒")
        
        async def watch_loop():
            while True:
                try:
                    await self._check_for_changes()
                    await asyncio.sleep(interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"配置监控循环错误: {e}")
                    self._error_count += 1
                    self._last_error_time = time.time()
                    
                    # 错误过多时暂停
                    if self._error_count > 10 and time.time() - self._last_error_time < 60:
                        print("配置监控错误过多，暂停60秒")
                        await asyncio.sleep(60)
                    else:
                        await asyncio.sleep(5)
        
        self._watch_task = asyncio.create_task(watch_loop())
    
    async def _check_for_changes(self):
        """检查配置变化"""
        if not self.config_path.exists():
            print(f"配置文件不存在: {self.config_path}")
            return
        
        try:
            # 检查文件修改时间
            current_mtime = self.config_path.stat().st_mtime
            if current_mtime <= self._last_modified:
                return
            
            # 读取配置文件
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    print("配置文件为空")
                    return
                
                new_config_data = json.loads(content)
            
            # 计算新哈希
            new_hash = self._calculate_config_hash(new_config_data)
            
            # 如果哈希值相同，只是时间戳变化
            if new_hash == self._last_hash:
                self._last_modified = current_mtime
                return
            
            print(f"检测到配置变化，重新加载配置")
            
            # 重新加载配置
            await self._reload_config(new_config_data)
            
            # 更新状态
            self._last_hash = new_hash
            self._last_modified = current_mtime
            self._error_count = 0
            
        except json.JSONDecodeError as e:
            print(f"配置文件JSON格式错误: {e}")
        except Exception as e:
            print(f"检查配置变化失败: {e}")
            self._error_count += 1
    
    async def _reload_config(self, new_config_data: Dict[str, Any]):
        """重新加载配置"""
        try:
            async with self._lock:
                old_config = self.current_config
                
                # 创建新配置实例
                new_config = PluginConfig(**new_config_data)
                
                # 验证新配置
                if not self._validate_config(new_config):
                    print("新配置验证失败，保持当前配置")
                    return
                
                # 通知监听器（在更新当前配置之前）
                print(f"通知 {len(self._listeners)} 个监听器配置变更")
                
                listener_tasks = []
                for listener in self._listeners:
                    try:
                        if asyncio.iscoroutinefunction(listener):
                            task = asyncio.create_task(listener(old_config, new_config))
                            listener_tasks.append(task)
                        else:
                            # 同步函数在线程池中执行
                            loop = asyncio.get_event_loop()
                            task = loop.run_in_executor(None, listener, old_config, new_config)
                            listener_tasks.append(task)
                    except Exception as e:
                        print(f"配置变更监听器调用失败: {e}")
                
                # 等待所有监听器完成
                if listener_tasks:
                    try:
                        await asyncio.gather(*listener_tasks, return_exceptions=True)
                    except Exception as e:
                        print(f"等待监听器完成时出错: {e}")
                
                # 更新当前配置
                self.current_config = new_config
                
                print("配置热重载完成")
                
                # 记录配置差异
                self._log_config_changes(old_config, new_config)
                
        except Exception as e:
            print(f"配置重载失败: {e}")
            raise
    
    def _validate_config(self, config: PluginConfig) -> bool:
        """验证配置的有效性"""
        try:
            # 检查基本约束
            if config.favour_min >= config.favour_max:
                print(f"配置验证失败: favour_min ({config.favour_min}) >= favour_max ({config.favour_max})")
                return False
            
            if config.intimacy_min >= config.intimacy_max:
                print(f"配置验证失败: intimacy_min ({config.intimacy_min}) >= intimacy_max ({config.intimacy_max})")
                return False
            
            if config.change_min >= config.change_max:
                print(f"配置验证失败: change_min ({config.change_min}) >= change_max ({config.change_max})")
                return False
            
            if config.force_update_interval <= 0:
                print(f"配置验证失败: force_update_interval ({config.force_update_interval}) <= 0")
                return False
            
            # 检查管理员列表格式
            for qq in config.admin_qq_list:
                if not isinstance(qq, str) or not qq.isdigit():
                    print(f"配置验证失败: 无效的管理员QQ号格式: {qq}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"配置验证过程中出错: {e}")
            return False
    
    def _log_config_changes(self, old_config: PluginConfig, new_config: PluginConfig):
        """记录配置变化"""
        changes = []
        
        old_dict = old_config.dict()
        new_dict = new_config.dict()
        
        for key in old_dict.keys():
            if key in new_dict and old_dict[key] != new_dict[key]:
                changes.append({
                    'key': key,
                    'old': old_dict[key],
                    'new': new_dict[key]
                })
        
        if changes:
            print("配置变化详情:")
            for change in changes:
                print(f"  {change['key']}: {change['old']} -> {change['new']}")
        else:
            print("没有检测到配置值变化")
    
    async def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        async with self._lock:
            try:
                old_config = self.current_config
                config_dict = old_config.dict()
                
                # 应用更新
                config_dict.update(updates)
                
                # 创建新配置实例
                new_config = PluginConfig(**config_dict)
                
                # 验证新配置
                if not self._validate_config(new_config):
                    raise ValueError("新配置验证失败")
                
                # 保存到文件
                self._save_config(new_config)
                
                # 通知监听器
                listener_tasks = []
                for listener in self._listeners:
                    try:
                        if asyncio.iscoroutinefunction(listener):
                            task = asyncio.create_task(listener(old_config, new_config))
                            listener_tasks.append(task)
                        else:
                            loop = asyncio.get_event_loop()
                            task = loop.run_in_executor(None, listener, old_config, new_config)
                            listener_tasks.append(task)
                    except Exception as e:
                        print(f"配置变更监听器调用失败: {e}")
                
                # 等待所有监听器完成
                if listener_tasks:
                    await asyncio.gather(*listener_tasks, return_exceptions=True)
                
                # 更新当前配置
                self.current_config = new_config
                
                print(f"配置更新成功: {len(updates)} 个字段已更新")
                
                return True
                
            except Exception as e:
                print(f"更新配置失败: {e}")
                return False
    
    async def get_config_snapshot(self) -> Dict[str, Any]:
        """获取配置快照"""
        async with self._lock:
            config_dict = self.current_config.dict()
            
            return {
                'config': config_dict,
                'metadata': {
                    'config_path': str(self.config_path),
                    'last_modified': self._last_modified,
                    'config_hash': self._last_hash,
                    'listener_count': len(self._listeners),
                    'error_count': self._error_count
                }
            }
    
    async def stop_watching(self):
        """停止监控"""
        if self._watch_task:
            print("停止配置监控任务...")
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None
            print("配置监控任务已停止")
    
    async def refresh(self):
        """手动刷新配置"""
        print("手动刷新配置...")
        await self._check_for_changes()
    
    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        return {
            'watching': self._watch_task is not None,
            'listener_count': len(self._listeners),
            'error_count': self._error_count,
            'last_modified': self._last_modified,
            'config_file_exists': self.config_path.exists()
        }