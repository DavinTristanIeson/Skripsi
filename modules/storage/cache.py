from collections import OrderedDict
from dataclasses import dataclass, field
import datetime
import threading
from typing import Callable, Generic, Optional, TypeVar

from modules.logger import ProvisionedLogger

T = TypeVar("T")

logger = ProvisionedLogger().provision("CacheClient")

@dataclass
class CacheItem(Generic[T]):
  key: str
  value: T
  persistent: bool = False
  cached_at: datetime.datetime = field(
    default_factory=lambda: datetime.datetime.now()
  )
  def is_stale(self, seconds: int)->bool:
    if self.persistent:
      return False
    delta = datetime.datetime.now() - self.cached_at
    return delta.seconds > seconds

@dataclass
class CacheClient(Generic[T]):
  name: str
  maxsize: Optional[int]
  ttl: Optional[int]
  lock: threading.RLock = field(default_factory=lambda: threading.RLock(), init=False)
  records: OrderedDict[str, CacheItem[T]] = field(default_factory=lambda: OrderedDict(), init=False)

  def get(self, key: str)->Optional[T]:
    with self.lock:
      value = self.records.get(key, None)
      if value is None:
        logger.debug(f"[{self.name}] GET {key} (CACHE MISS)")
        return None
      if self.ttl is not None and value.is_stale(self.ttl):
        logger.debug(f"[{self.name}] GET {key} (CACHE STALE)")
        self.records.pop(key, None)
        return None
      # LRU
      logger.debug(f"[{self.name}] GET {key} (CACHE HIT)")
      self.records.move_to_end(key)
      return value.value
    
  def get_or(self, key: str, factory: Callable[[], T])->T:
    value = self.get(key)
    if value is None:
      return factory()
    return value
  
  def pop_lru(self):
    if self.maxsize is None:
      return
    total_count = len(self.records)
    if total_count <= self.maxsize:
      return
    with self.lock:
      targets: list[str] = []
      for cache_key, cache_value in self.records.items():
        if cache_value.persistent:
          continue
        targets.append(cache_key)
        if total_count - len(targets) <= self.maxsize:
          break
      logger.debug(f"[{self.name}] POP LRU: {targets}")
      for target in targets:
        self.records.pop(target)

  def set(self, value: CacheItem[T]):
    self.records[value.key] = value
    logger.debug(f"[{self.name}] SET {value.key}")
    if len(self.records) == 0:
      return
    self.pop_lru()

  def invalidate(self, *, key: Optional[str] = None, prefix: Optional[str] = None):
    with self.lock:
      if key is not None:
        logger.debug(f"[{self.name}] INVALIDATE {key}")
        self.records.pop(key, None)
      if prefix is not None:
        cache_keys = [cache_key for cache_key in self.records.keys() if cache_key.startswith(prefix)]
        logger.debug(f"[{self.name}] INVALIDATE {', '.join(cache_keys)}")
        for cache_key in cache_keys:
          self.records.pop(cache_key, None)

  def clear(self):
    with self.lock:
      self.records.clear()

__all__ = [
  "CacheClient",
  "CacheItem"
]