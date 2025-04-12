from typing import Any, Optional
import pydantic

class DashboardItemRect(pydantic.BaseModel):
  x: int = pydantic.Field(ge=0)
  y: int = pydantic.Field(ge=0)
  width: int = pydantic.Field(gt=0)
  height: int = pydantic.Field(gt=0)

class DashboardItem(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
  id: str
  title: str
  description: Optional[str]
  column: str
  rect: DashboardItemRect
  # FE deals with validating this.
  config: dict[str, Any]

class Dashboard(pydantic.BaseModel):
  items: list[DashboardItem]

__all__ = [
  "DashboardItem",
]