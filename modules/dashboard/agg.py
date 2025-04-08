from typing import Annotated, Union

import pydantic

from modules.validation.exceptions import DiscriminatedUnionValidator
from .continuous import ContinuousDashboardItemUnion

DashboardItem = Annotated[
  ContinuousDashboardItemUnion,
  pydantic.Field(discriminator="type"),
  DiscriminatedUnionValidator
]

class Dashboard(pydantic.BaseModel):
  items: list[DashboardItem]

__all__ = [
  "DashboardItem",
  "Dashboard"
]