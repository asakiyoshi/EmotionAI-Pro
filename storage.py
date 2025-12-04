# storage.py
import json
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, Any, Optional, Set, List, Tuple
from dataclasses import asdict
import shutil
from datetime import datetime, timedelta
import hashlib

from .models import EnhancedEmotionalState
from .constants import PathConstants

class AtomicJSONStorage:
    """原子JSON存储 - 避免数据损坏"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.temp_file_path = file_path.with_suffix('.tmp')
        self.lock = asyncio.Lock()
        self._checksum_file = file_path.with_suffix('.checksum')
    
    async def load(self) -> Dict[str, Any]:
        """原子加载数据 - 增强版本，包含校验和验证"""
        async with self.lock:
            if not self.file_path.exists():
                return {}
            
            # 尝试加载主文件
            try:
                async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    
                    # 验证文件完整性
                    if await self._verify_checksum(content):
                        return json.loads(content) if content.strip() else {}
                    else:
                        print(f"警告: 文件 {self.file_path} 校验和不匹配")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"主文件加载失败: {e}")
            
            # 主文件损坏，尝试备份文件
            backup_path = self.file_path.with_suffix('.bak')
            if backup_path.exists():
                try:
                    async with aiofiles.open(backup_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        data = json.loads(content) if content.strip() else {}
                        
                        # 恢复备份
                        await self._save_data(data)
                        print(f"从备份文件恢复数据: {backup_path}")
                        return data
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"备份文件也损坏: {e}")
            
            # 两个文件都损坏，返回空数据
            print(f"警告: 数据文件 {self.file_path} 完全损坏，返回空数据")
            return {}
    
    async def save(self, data: Dict[str, Any]):
        """原子保存数据 - 增强版本，包含校验和"""
        async with self.lock:
            await self._save_data(data)
    
    async def _save_data(self, data: Dict[str, Any]):
        """实际保存数据（带备份和校验和）"""
        # 确保目录存在
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 序列化数据
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        
        # 计算校验和
        checksum = self._calculate_checksum(json_str)
        
        # 如果原文件存在，先备份
        if self.file_path.exists():
            backup_path = self.file_path.with_suffix('.bak')
            try:
                shutil.copy2(self.file_path, backup_path)
                print(f"创建备份文件: {backup_path}")
            except Exception as e:
                print(f"备份创建失败: {e}")
        
        # 先写入临时文件
        try:
            async with aiofiles.open(self.temp_file_path, 'w', encoding='utf-8') as f:
                await f.write(json_str)
            
            # 保存校验和
            await self._save_checksum(checksum)
            
            # 原子重命名
            self.temp_file_path.replace(self.file_path)
            print(f"成功保存数据到: {self.file_path}")
            
        except Exception as e:
            # 如果失败，删除临时文件
            if self.temp_file_path.exists():
                self.temp_file_path.unlink()
            print(f"数据保存失败: {e}")
            raise e
    
    def _calculate_checksum(self, content: str) -> str:
        """计算内容的校验和"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    async def _save_checksum(self, checksum: str):
        """保存校验和"""
        try:
            async with aiofiles.open(self._checksum_file, 'w', encoding='utf-8') as f:
                await f.write(checksum)
        except Exception as e:
            print(f"校验和保存失败: {e}")
    
    async def _verify_checksum(self, content: str) -> bool:
        """验证校验和"""
        if not self._checksum_file.exists():
            return True  # 如果没有校验和文件，跳过验证
        
        try:
            async with aiofiles.open(self._checksum_file, 'r', encoding='utf-8') as f:
                saved_checksum = await f.read()
            
            current_checksum = self._calculate_checksum(content)
            return saved_checksum.strip() == current_checksum
        except Exception:
            return False

class UserStateRepository:
    """用户状态仓库 - 修复版本：差异化更新用户数据"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.user_storage = AtomicJSONStorage(data_dir / PathConstants.USER_DATA_FILE)
        self.memory_storage = AtomicJSONStorage(data_dir / PathConstants.LONG_TERM_MEMORY_FILE)
        
        self._user_data: Dict[str, Dict[str, Any]] = {}
        self._memory_data: Dict[str, Any] = {}
        self._loaded = False
        self._lock = asyncio.Lock()  # 添加锁，防止并发问题
    
    async def load_all_data(self):
        """加载所有数据"""
        async with self._lock:
            if not self._loaded:
                self._user_data = await self.user_storage.load()
                self._memory_data = await self.memory_storage.load()
                self._loaded = True
    
    async def get_user_state(self, user_key: str) -> Optional[EnhancedEmotionalState]:
        """获取用户状态"""
        await self.load_all_data()
        
        if user_key in self._user_data:
            try:
                return EnhancedEmotionalState.from_dict(self._user_data[user_key])
            except (TypeError, KeyError, ValueError) as e:
                print(f"用户 {user_key} 数据格式错误: {e}")
                # 尝试修复损坏的数据
                await self._try_repair_user_data(user_key)
                return None
        return None
    
    async def _try_repair_user_data(self, user_key: str):
        """尝试修复损坏的用户数据"""
        async with self._lock:
            if user_key in self._user_data:
                try:
                    # 创建一个默认状态
                    default_state = EnhancedEmotionalState(user_key=user_key)
                    self._user_data[user_key] = default_state.to_dict()
                    await self.user_storage.save(self._user_data)
                    print(f"已修复用户 {user_key} 的损坏数据")
                except Exception as e:
                    print(f"修复用户数据失败: {e}")
    
    async def save_user_state(self, user_key: str, state: EnhancedEmotionalState):
        """保存用户状态 - 差异化更新"""
        await self.load_all_data()
        
        async with self._lock:
            # 只更新特定用户的数据
            self._user_data[user_key] = state.to_dict()
            await self.user_storage.save(self._user_data)
    
    async def save_all_user_states(self, states: Dict[str, EnhancedEmotionalState]):
        """批量保存用户状态 - 修复版本：只更新有变化的用户"""
        await self.load_all_data()
        
        async with self._lock:
            # 只更新传入的用户状态，保留其他用户的数据
            updated_count = 0
            for key, state in states.items():
                # 只有当状态确实发生变化时才更新
                current_state_dict = self._user_data.get(key, {})
                new_state_dict = state.to_dict()
                
                # 使用更精确的比较
                if self._is_state_different(current_state_dict, new_state_dict):
                    self._user_data[key] = new_state_dict
                    updated_count += 1
            
            if updated_count > 0:
                await self.user_storage.save(self._user_data)
                print(f"差异化保存: 更新了 {updated_count} 个用户状态，总用户数: {len(self._user_data)}")
            else:
                print("没有需要保存的用户状态变更")
    
    def _is_state_different(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> bool:
        """比较两个状态是否不同"""
        # 简化比较，只比较关键字段
        key_fields = ['favor', 'intimacy', 'relationship_stage', 'force_update_counter']
        
        for field in key_fields:
            if old_state.get(field) != new_state.get(field):
                return True
        
        # 比较互动统计
        old_stats = old_state.get('stats', {})
        new_stats = new_state.get('stats', {})
        if old_stats.get('total_count') != new_stats.get('total_count'):
            return True
        
        return False
    
    async def save_updated_user_states_only(self, updated_states: Dict[str, EnhancedEmotionalState]):
        """只保存已更新的用户状态（更轻量级的方法）"""
        await self.load_all_data()
        
        async with self._lock:
            # 更新内存中的数据
            for user_key, state in updated_states.items():
                self._user_data[user_key] = state.to_dict()
            
            # 保存到文件
            await self.user_storage.save(self._user_data)
            print(f"保存了 {len(updated_states)} 个更新的用户状态")
    
    async def get_all_user_states(self) -> Dict[str, EnhancedEmotionalState]:
        """获取所有用户状态"""
        await self.load_all_data()
        
        result = {}
        for user_key, data in self._user_data.items():
            try:
                result[user_key] = EnhancedEmotionalState.from_dict(data)
            except (TypeError, KeyError, ValueError) as e:
                print(f"用户 {user_key} 数据格式错误: {e}")
                # 跳过损坏的数据
                continue
        return result
    
    async def delete_user_state(self, user_key: str) -> bool:
        """删除用户状态"""
        await self.load_all_data()
        
        async with self._lock:
            if user_key in self._user_data:
                del self._user_data[user_key]
                await self.user_storage.save(self._user_data)
                return True
        return False
    
    async def get_memory_data(self) -> Dict[str, Any]:
        """获取记忆数据"""
        await self.load_all_data()
        return self._memory_data.copy()
    
    async def save_memory_data(self, data: Dict[str, Any]):
        """保存记忆数据"""
        await self.load_all_data()
        
        async with self._lock:
            self._memory_data = data
            await self.memory_storage.save(self._memory_data)
    
    async def get_user_count(self) -> int:
        """获取用户数量"""
        await self.load_all_data()
        return len(self._user_data)

class BackupManager:
    """备份管理器 - 修复版本"""
    
    def __init__(self, data_dir: Path, retention_days: int = 7):
        self.data_dir = data_dir
        self.backup_dir = data_dir / PathConstants.BACKUP_DIR
        self.backup_dir.mkdir(exist_ok=True)
        self.retention_days = retention_days
        self._lock = asyncio.Lock()
    
    async def create_backup(self) -> str:
        """创建备份 - 修复参数问题"""
        async with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"emotionai_backup_{timestamp}"
            backup_path = self.backup_dir / backup_name
            
            backup_path.mkdir(exist_ok=True)
            
            # 复制数据文件
            files_to_backup = [
                PathConstants.USER_DATA_FILE,
                PathConstants.LONG_TERM_MEMORY_FILE
            ]
            
            backed_up_files = 0
            for filename in files_to_backup:
                src = self.data_dir / filename
                if src.exists():
                    dst = backup_path / filename
                    try:
                        shutil.copy2(src, dst)
                        backed_up_files += 1
                    except Exception as e:
                        print(f"备份文件 {filename} 失败: {e}")
            
            if backed_up_files == 0:
                raise Exception("没有文件可备份")
            
            # 创建备份元数据
            await self._create_backup_metadata(backup_path, backed_up_files)
            
            # 清理旧备份
            await self._cleanup_old_backups()
            
            print(f"备份创建成功: {backup_name}, 包含 {backed_up_files} 个文件")
            return backup_name
    
    async def _create_backup_metadata(self, backup_path: Path, file_count: int):
        """创建备份元数据文件"""
        metadata = {
            'backup_time': datetime.now().isoformat(),
            'file_count': file_count,
            'data_dir': str(self.data_dir),
            'plugin_version': '4.0.0'
        }
        
        metadata_path = backup_path / 'backup_metadata.json'
        async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(metadata, ensure_ascii=False, indent=2))
    
    async def _cleanup_old_backups(self):
        """清理旧备份"""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir() and backup_dir.name.startswith("emotionai_backup_"):
                try:
                    # 从目录名解析时间
                    time_str = backup_dir.name.replace("emotionai_backup_", "")
                    dir_time = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
                    
                    if dir_time < cutoff_time:
                        shutil.rmtree(backup_dir)
                        print(f"删除旧备份: {backup_dir.name}")
                except ValueError:
                    # 目录名格式错误，检查元数据
                    metadata_path = backup_dir / 'backup_metadata.json'
                    if metadata_path.exists():
                        try:
                            async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.loads(await f.read())
                                backup_time = datetime.fromisoformat(metadata['backup_time'])
                                if backup_time < cutoff_time:
                                    shutil.rmtree(backup_dir)
                                    print(f"通过元数据删除旧备份: {backup_dir.name}")
                        except Exception:
                            # 无法读取元数据，跳过
                            continue
    
    async def list_backups(self) -> List[Dict[str, Any]]:
        """列出所有备份的详细信息"""
        backups = []
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir() and backup_dir.name.startswith("emotionai_backup_"):
                backup_info = {
                    'name': backup_dir.name,
                    'path': str(backup_dir),
                    'size': self._get_dir_size(backup_dir),
                    'files': []
                }
                
                # 添加文件列表
                for file_path in backup_dir.iterdir():
                    if file_path.is_file():
                        backup_info['files'].append({
                            'name': file_path.name,
                            'size': file_path.stat().st_size
                        })
                
                # 尝试获取元数据
                metadata_path = backup_dir / 'backup_metadata.json'
                if metadata_path.exists():
                    try:
                        async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.loads(await f.read())
                            backup_info['metadata'] = metadata
                    except Exception:
                        pass
                
                backups.append(backup_info)
        
        return sorted(backups, key=lambda x: x['name'], reverse=True)
    
    def _get_dir_size(self, directory: Path) -> int:
        """计算目录大小"""
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    async def restore_backup(self, backup_name: str) -> bool:
        """从备份恢复数据"""
        backup_path = self.backup_dir / backup_name
        if not backup_path.exists():
            return False
        
        try:
            # 备份当前数据
            await self.create_backup()
            
            # 恢复文件
            for filename in [PathConstants.USER_DATA_FILE, PathConstants.LONG_TERM_MEMORY_FILE]:
                src = backup_path / filename
                dst = self.data_dir / filename
                
                if src.exists():
                    shutil.copy2(src, dst)
                    print(f"恢复文件: {filename}")
            
            print(f"成功从备份 {backup_name} 恢复数据")
            return True
            
        except Exception as e:
            print(f"恢复备份失败: {e}")
            return False