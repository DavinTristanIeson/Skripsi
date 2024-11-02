import hashlib
from typing import Generator, Generic, Iterable, Sequence, TypeVar

def hashfile(path: str)->str:
  hasher = hashlib.md5()
  with open(path, 'rb') as f:
    while (chunk := f.read(4096)):
      hasher.update(chunk)
  return hasher.hexdigest()

def concatenate_generator(iterable: Iterable[Sequence[str]], separator: str = ' ')->Iterable[str]:
  return map(lambda tokens: separator.join(tokens), iterable)

TIter = TypeVar("TIter")
TReturn = TypeVar("TReturn")

class GeneratorValueCatcher(Generic[TIter, TReturn]):
  # https://stackoverflow.com/questions/34073370/best-way-to-receive-the-return-value-from-a-python-generator
  def __init__(self, generator: Generator[TIter, None, TReturn]):
    self.generator = generator
  def __iter__(self):
    self.value = yield from self.generator
    return self.value