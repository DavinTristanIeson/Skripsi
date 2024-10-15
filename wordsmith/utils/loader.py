import hashlib
from typing import Iterable, Sequence, TextIO

TOKEN_DELIMITER = ' '
SENTENCE_DELIMITER = '\n'

def write_tokens(f: TextIO, batch: Iterable[Sequence[str]])->None:
  f.write(SENTENCE_DELIMITER.join(TOKEN_DELIMITER.join(line) for line in batch if len(line) > 0))

def load_lines(f: TextIO, *, repeat: bool = False)->Iterable[str]:
  while True:
    f.seek(0)
    while line:=f.readline():
      yield line.strip()
    if not repeat:
      break
    
def load_tokens(f: TextIO, *, repeat: bool = False)->Iterable[Sequence[str]]:
  generator = load_lines(f, repeat=repeat)
  return (line.strip().split(TOKEN_DELIMITER) for line in generator)

def hashfile(path: str)->str:
  hasher = hashlib.md5()
  with open(path, 'rb') as f:
    while (chunk := f.read(4096)):
      hasher.update(chunk)
  return hasher.hexdigest()