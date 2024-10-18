from typing import Callable, Iterable, Optional, Sequence, TypeVar
import numpy.typing as npt

T = TypeVar("T")
def select_indexes(arr: Sequence[T], indices: Iterable[int])->Iterable[T]:
  return map(lambda i: arr[i], indices)

def apply_mask(arr: Sequence[T], mask: npt.NDArray)->Iterable[T]:
  indices = mask.nonzero()[0]
  return (arr[idx] for idx in indices)

def array_find(arr: Iterable[T], fn: Callable[[T], bool])->Optional[T]:
  for item in arr:
    if fn(item):
      return item
  return None


T = TypeVar("T")
def batched(data: Iterable[T], batch_size: int)->Iterable[Sequence[T]]:
  buffer: list[T] = []
  for item in data:
    buffer.append(item)
    if len(buffer) > batch_size:
      yield tuple(buffer) # Create copy.
      buffer.clear()
  yield tuple(buffer)

def flatten(data: Iterable[Iterable[T]])->Iterable[T]:
  for line in data:
    for word in line:
      yield word