from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, cast

from common.models.metaclass import Singleton

@dataclass
class EnumMemberDescriptor:
  label: str
  description: Optional[str] = None

@dataclass
class EnumDescriptor:
  cls: Any
  labels: dict[Any, EnumMemberDescriptor]

  def value_mapping(self)->dict[str, str]:
    field = dict()
    for enum_member in self.cls.__members__.values():
      field[cast(Enum, enum_member).name] = cast(Enum, enum_member).value
    return field

class ExposedEnum(metaclass=Singleton):
  registrar: dict[str, EnumDescriptor]
  def __init__(self):
    self.registrar = {}

  def register(self, enum: Any, labels: dict[Any, EnumMemberDescriptor]):
    self.registrar[enum.__name__] = EnumDescriptor(
      cls=enum,
      labels=labels
    )

  def get_all_enums(self)->dict[str, dict[str, str]]:
    members = {}
    for name, enum_meta in self.registrar.items():
      members[name] = enum_meta.value_mapping()
    return members
  
  def get_enum(self, enum: str)->Optional[EnumDescriptor]:
    enum_meta = self.registrar.get(enum, None)
    if enum_meta:
      return enum_meta
    return None
