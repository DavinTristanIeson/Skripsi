from typing import Iterable, Sequence

def truncate_strings(labels: Sequence[str], limit: int = 30)->Iterable[str]:
  return map(
    lambda label: label[:(limit-3)] + '...' if len(label) > limit else label,
    labels,
  )

def concatenate_generator(iterable: Iterable[Sequence[str]], separator: str = ' ')->Iterable[str]:
  return map(lambda tokens: separator.join(tokens), iterable)
