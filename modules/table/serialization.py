from typing import Any
import pandas as pd

def serialize_pandas(data: pd.DataFrame)->list[dict[str, Any]]:
  # Stupid way, but it's necessary to deal with serializing NaNs and NaTs.
  import orjson
  json_response = data.to_json(orient="records")
  serialized_data = orjson.loads(json_response)
  return serialized_data

__all__ = ["serialize_pandas"]