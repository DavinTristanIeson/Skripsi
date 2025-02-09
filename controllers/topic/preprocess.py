from dataclasses import dataclass
import os
import pandas as pd
from typing import Sequence

from common.models.api import ApiError
from common.task.executor import TaskPayload
from controllers.topic.utils import BERTopicColumnIntermediateResult
from models.config.paths import ProjectPaths
from models.config.schema import TextualSchemaColumn
from models.project.cache import ProjectCacheManager, get_cached_data_source

def bertopic_load_workspace(task: TaskPayload):
  cache = ProjectCacheManager().get(task.request.project_id)
  config = cache.config

  result_path = config.paths.full_path(ProjectPaths.Workspace)
  if os.path.exists(result_path):
    task.progress(f"Loading cached dataset from \"{result_path}\"")
    df = cache.load_workspace()
  else:
    task.progress(f"Loading dataset from \"{config.source.path}\"")
    raw_df = get_cached_data_source(config.source)
    df = config.data_schema.fit(raw_df)
    task.progress(f"Saving dataset from \"{config.source.path}\" to \"{result_path}\"")
    df.to_parquet(result_path)
  return df

def bertopic_preprocessing(
  df: pd.DataFrame,
  column: TextualSchemaColumn,
  task: TaskPayload,
)->BERTopicColumnIntermediateResult:
  raw_documents = df[column.name]
  mask = raw_documents.notna()
  original_documents: Sequence[str] = raw_documents[mask] # type: ignore

  task.progress(f"Preprocessing the documents in column \"{column.name}\". Text preprocessing may take some time...")
  if column.preprocess_column.name not in df.columns:
    df.loc[mask, column.preprocess_column] = column.preprocessing.preprocess_topic_keywords(original_documents) # type: ignore

  sbert_documents = column.preprocessing.preprocess_sbert(original_documents)

  return BERTopicColumnIntermediateResult(
    column=column,
    documents=sbert_documents,
    mask=mask,
    embeddings=None, # type: ignore
  )


    
    