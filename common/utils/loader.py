import hashlib
from typing import Iterable, Sequence

def hashfile(path: str)->str:
  hasher = hashlib.md5()
  with open(path, 'rb') as f:
    while (chunk := f.read(4096)):
      hasher.update(chunk)
  return hasher.hexdigest()

def concatenate_generator(iterable: Iterable[Sequence[str]], separator: str = ' ')->Iterable[str]:
  return map(lambda tokens: separator.join(tokens), iterable)