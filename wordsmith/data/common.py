from enum import Enum
import os
import nltk

NLTK_MODEL_PATH = "models/nltk"
# Force use local path
nltk.data.path.clear()
nltk.data.path.append(NLTK_MODEL_PATH)


class PreprocessingModuleConfigException(Exception):
  pass

class DataSourceLoadingException(Exception):
  pass

class SupportedLanguageEnum(str, Enum):
  English = "english"
  Indonesian = "indonesian"


INTERMEDIATE_DIRECTORY = "intermediate"
class RelevantDataPaths(str, Enum):
  # Preprocessing results
  IntermediateDirectory = INTERMEDIATE_DIRECTORY
  StemmingDictionary = os.path.join(INTERMEDIATE_DIRECTORY, "stemming.json")
  GensimDictionary = os.path.join(INTERMEDIATE_DIRECTORY, "dictionary.dat")
  Table = os.path.join(INTERMEDIATE_DIRECTORY, "table.parquet")
  DataSourceHash = os.path.join(INTERMEDIATE_DIRECTORY, "source.hash.txt")
  Config = "config.json"

  # Topic modeling / cross-tabulation results
  ResultDirectory = "results"
  ResultTable = 'results/table.parquet'
  Topics = 'results/topics.parquet'
  DataReport = 'results/report.html'
  TopicReport = 'results/topics-report.html'

__all__ = [
  "PreprocessingModuleConfigException",
  "DataSourceLoadingException",
  "NLTK_MODEL_PATH",
  "SupportedLanguageEnum",
  "RelevantDataPaths"
]