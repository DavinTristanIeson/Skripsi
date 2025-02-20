import pandas as pd
from typing import Sequence

from .utils import BERTopicColumnIntermediateResult

def bertopic_preprocessing(
  df: pd.DataFrame,
  intermediate: BERTopicColumnIntermediateResult
):
  column = intermediate.column
  preprocess_name = column.preprocess_column.name
  raw_documents = df[column.name]
  if column.preprocess_column.name in df.columns:
    # CACHED
    raw_preprocess_documents = df[preprocess_name]
    mask = df[preprocess_name].notna()
    preprocess_documents = raw_preprocess_documents[mask]
  else:
    # COMPUTE
    original_mask = raw_documents.notna()
    original_documents: Sequence[str] = raw_documents[original_mask] # type: ignore

    intermediate.task.log_pending(f"Preprocessing the documents in column \"{column.name}\". Text preprocessing may take some time...")
    # preprocess_topic_keywords set NA for invalid documents, so we need to recompute mask
    df.loc[original_mask, preprocess_name] = column.preprocessing.preprocess_topic_keywords(original_documents) # type: ignore
    mask = df[preprocess_name].notna()
    preprocess_documents = df.loc[mask, preprocess_name]
    intermediate.task.log_success(f"Finished preprocessing the documents in column \"{column.name}\".")
  
  if len(preprocess_documents) == 0:
    raise ValueError(f"\"{column.name}\" does not contain any valid documents after the preprocessing step. Either change the preprocessing configuration of \"{column.name}\" to be more lax (e.g: lower the min word frequency, min document length), or set the type of this column to Unique.")
  
  intermediate.mask = mask
  original_documents: Sequence[str] = raw_documents[mask] # type: ignore
  # Light preprocessing for SBERT
  intermediate.task.log_pending(f"Performing light preprocessing for the documents in column \"{column.name}\". This shouldn't take too long...")
  sbert_documents = column.preprocessing.preprocess_sbert(original_documents)
  intermediate.task.log_pending(f"Finished performing light preprocessing for the documents in column \"{column.name}\". {len(original_documents) - len(preprocess_documents)} document(s) has been excluded from the topic modeling process.")

  intermediate.embedding_documents = sbert_documents
  intermediate.documents = preprocess_documents # type: ignore



    
    