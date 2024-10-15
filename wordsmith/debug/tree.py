
# REFERENCE: https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
from dataclasses import dataclass
from typing import Iterable, Sequence


class TreePrettyPrintComponents:
  # prefix components:
  SPACE =  '    '
  BRANCH = '│   '
  # pointers:
  TEE =    '├── '
  LAST =   '└── '

@dataclass
class PrintableTreeStructure:
  label: str
  children: Sequence["PrintableTreeStructure"]

  @property
  def traversable(self)->bool:
    return len(self.children) > 0


def logtree_generator(tree: PrintableTreeStructure, prefix: str=''):
  """A recursive generator, given a PrintableTreeStructure
  will yield a visual tree structure line by line
  with each line prefixed by the same characters
  """    
  # contents each get pointers that are ├── with a final └── :
  pointers = [TreePrettyPrintComponents.TEE] * (len(tree.children) - 1) + [TreePrettyPrintComponents.LAST]
  for idx, child in enumerate(tree.children):
    if idx == len(tree.children) - 1:
      pointer = TreePrettyPrintComponents.LAST
    else:
      pointer = TreePrettyPrintComponents.TEE
    yield prefix + pointer + child.label
    if child.traversable: # extend the prefix and recurse:
      extension = TreePrettyPrintComponents.BRANCH \
        if pointer == TreePrettyPrintComponents.TEE \
        else TreePrettyPrintComponents.SPACE
      # i.e. space because last, └── , above so no more |
      yield from logtree_generator(child, prefix=prefix + extension)

def logtree(tree: PrintableTreeStructure)->str:
  return '\n'.join(logtree_generator(tree))
    
