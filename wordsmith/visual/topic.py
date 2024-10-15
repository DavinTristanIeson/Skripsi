from typing import Optional, Sequence, Union
import numpy as np
import plotly.graph_objects as go
import plotly.offline
import plotly.subplots
import pandas as pd


def as_barcharts(topics: Union[Sequence[Sequence[tuple[str, float]]], pd.Series], *, title: str, labels: Optional[Sequence[str]] = None, height: int = 250):
  '''Copied from bertopic/plotting/_barchart.'''
  
  subplot_titles: Sequence[str]
  if labels is not None:
    subplot_titles = labels
  else:
    subplot_titles = tuple(map(lambda x: f"Topic {x}", range(len(topics))))

  columns = 4
  rows = int(np.ceil(len(topics) / columns))
  fig = plotly.subplots.make_subplots(
      rows=rows,
      cols=columns,
      shared_xaxes=False,
      horizontal_spacing=0.1,
      vertical_spacing=0.4 / rows if rows > 1 else 0,
      subplot_titles=subplot_titles,
  )

  # Add barchart for each topic
  row = 1
  column = 1
  for topic in topics:
    if len(topic) == 0:
      continue

    words = [word + "  " for word, _ in topic][::-1]
    scores = [score for _, score in topic][::-1]
    fig.add_trace(
        go.Bar(x=scores, y=words, orientation="h"),
        row=row,
        col=column,
    )

    if len(words) > 12:
        height = 250 + (len(words) - 12) * 11

    if len(words) > 9:
        fig.update_yaxes(tickfont=dict(size=(height - 140) // len(words)))

    if column == columns:
        column = 1
        row += 1
    else:
        column += 1

  # Stylize graph
  fig.update_layout(
    showlegend=False,
    title={
      "text": title,
      "x": 0.5,
      "xanchor": "center",
      "yanchor": "top",
      "font": dict(size=22, color="Black"),
    },
    height=height * rows if rows > 1 else height * 1.3,
    hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
  )

  fig.update_xaxes(showgrid=True)
  fig.update_yaxes(showgrid=True)

  return fig