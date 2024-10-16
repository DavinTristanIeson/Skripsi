import hashlib

def hashfile(path: str)->str:
  hasher = hashlib.md5()
  with open(path, 'rb') as f:
    while (chunk := f.read(4096)):
      hasher.update(chunk)
  return hasher.hexdigest()