from enum import Enum


class ImportableEnum(Enum):
  @staticmethod
  def get_all_enums():
    members = {}
    for subcls in ImportableEnum.__subclasses__():
      field = {}
      for member in subcls.__members__.values():
        field[member.name] = member.value
      members[str(subcls.name) + 'Enum'] = field
    return members
