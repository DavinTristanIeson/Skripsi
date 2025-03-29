from dataclasses import dataclass

@dataclass
class ConfigSerializationContext:
  is_save: bool

__all__ = [
  "ConfigSerializationContext"
]