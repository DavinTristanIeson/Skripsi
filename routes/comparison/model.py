
import pydantic
from modules.table import NamedTableFilter

# Schema
class CompareSubdatasetsSchema(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  column: str

class SubdatasetCooccurrenceResource(pydantic.BaseModel):
  labels: list[str]
  cooccurrences: list[list[int]]
  frequencies: list[int]

