from typing import Iterable, Optional, Sequence, TypeVar, cast


GensimBOW = Sequence[tuple[int,int]]
GensimBOWCorpus = Sequence[Sequence[tuple[int,int]]]
UntranslatedGensimTopics = Sequence[Sequence[tuple[int, float]]]
Topic = Sequence[tuple[str, float]]
GensimDocumentTopicDistribution = Sequence[tuple[int, float]]

T = TypeVar("T")

def notnone(data: Optional[T])->T:
  if data is None:
    raise ValueError("Value is not None")
  return cast(T, data)