# error_handling.py
import asyncio
import time
import traceback
from typing import Callable, Any
from functools import wraps

class CircuitBreaker:
    """断路器模式"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """使用断路器调用函数"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """调用成功处理"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """调用失败处理"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"操作 {func.__name__} 第 {attempt + 1} 次失败, {current_delay}秒后重试: {e}")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"操作 {func.__name__} 最终失败: {e}")
                        raise last_exception
            
            raise last_exception
        return wrapper
    return decorator

def error_handler(logger):
    """错误处理装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_msg = f"函数 {func.__name__} 执行失败: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                # 可以在这里发送错误通知
                raise e
        return wrapper
    return decorator