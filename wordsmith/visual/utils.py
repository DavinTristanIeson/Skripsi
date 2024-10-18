import math
from typing import Iterable, Sequence, Union
import plotly.graph_objects as go
import plotly.offline
import plotly.subplots
import re

def add_subplot(mainplot: go.Figure, subplot: go.Figure, coords: tuple[int, int]):
  for trace in subplot.data:
    mainplot.append_trace(trace, row=coords[0], col=coords[1])

def aggregate_subplot_figures(figures: Sequence[go.Figure], *, cols: int = 2, height: Union[int, Sequence[int]] = 400):
  rows_count = 0
  cols_count = cols
  rows_count = math.ceil(len(figures) / cols_count)

  HEIGHTS: Sequence[int] = (height,) * rows_count if isinstance(height, int) else height
  subplots = plotly.subplots.make_subplots(
    rows_count, cols_count,
    row_heights=HEIGHTS,
  )
  subplots.update_layout(
    height=sum(HEIGHTS),
  )

  row, col = 1, 1
  for fig in figures:
    add_subplot(subplots, fig, (row, col))
    if col == cols_count:
      col = 1
      row += 1
    else:
      col += 1
  return subplots