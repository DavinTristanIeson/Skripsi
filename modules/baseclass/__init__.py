from dataclasses import dataclass
from typing import Generic, TypeVar

class Singleton(type):
  _instances = {}
  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
    return cls._instances[cls]

T = TypeVar("T")

@dataclass
class ValueCarrier(Generic[T]):
  value: T