from typing import Annotated
import pydantic

FilenameField = Annotated[str, pydantic.Field(pattern = r"^[a-zA-Z0-9-_. ]+$", max_length = 255)]
FilePathField = Annotated[str, pydantic.Field(pattern = r"^[a-zA-Z0-9-_. \/\\:]+$")]

class __LinkValidationModel(pydantic.BaseModel):
  url: pydantic.AnyHttpUrl

def validate_http_url(url: str):
  if url is None or not isinstance(url, str) or len(url) == 0:
    return False
  try:
    __LinkValidationModel.model_validate(dict(url = url))
    return True
  except pydantic.ValidationError:
    return False

__all__ = [
  "validate_http_url",
  "FilenameField",
  "FilePathField"
]