from enum import Enum
from typing import Any, cast
from modules.baseclass import Singleton

class ExposedEnum(metaclass=Singleton):
  registrar: dict[str, Any]
  def __init__(self):
    self.registrar = {}

  def register(self, enum: Any):
    self.registrar[enum.__name__] = enum

  def get_all_enums(self)->dict[str, dict[str, str]]:
    members = {}
    for name, enum_meta in self.registrar.items():
      field = dict()
      for enum_member in enum_meta.__members__.values():
        enum_member = cast(Enum, enum_member)
        field[enum_member.name] = enum_member.value
      members[name] = field
    return members

__all__ = [
  "ExposedEnum"
]