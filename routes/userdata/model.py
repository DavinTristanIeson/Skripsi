
import pydantic
from modules.table.filter_variants import NamedTableFilter

# Validators
class ComparisonState(pydantic.BaseModel):
  groups: list[NamedTableFilter]

__all__ = [
  "ComparisonState",
]