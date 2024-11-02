import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy.typing as npt

def butterfly_chart(a: pd.DataFrame, b: pd.DataFrame, *, x: str, y: str):
  fig = go.Figure(
    [
      go.Bar(
        x=a[x] * -1,
        y=a[y],
        orientation='h',
        marker=dict(color="blue"),
        text=a[x],
      ),
      go.Bar(
        x=b[x],
        y=b[y],
        orientation='h',
        marker=dict(color="orange"),
      ),
    ],
    layout=go.Layout(
      xaxis=dict(title_text=x),
      yaxis=dict(title_text=y),
      barmode='overlay',
    )
  )
  return fig


def clustergram(df: pd.DataFrame):
  # https://plotly.com/python/dendrogram/

  # Upper dendrogram
  upper_dendrogram = ff.create_dendrogram(df, orientation="bottom", labels=df.columns)
  for i in range(len(upper_dendrogram['data'])): # type: ignore
    # Plotly internals is just a dictionary as config object
    upper_dendrogram['data'][i]['yaxis'] = 'y2'  # type: ignore
  
  # Side dendrogram
  side_dendrogram = ff.create_dendrogram(df, orientation="right", labels=df.columns)
  for i in range(len(side_dendrogram['data'])): # type: ignore
    side_dendrogram['data'][i]['yaxis'] = 'x2' # type: ignore

  side_dendro_leaves: list[str] = list(side_dendrogram.layout["yaxis"]["ticktext"])
  upper_dendro_leaves: list[str] = list(side_dendrogram.layout["yaxis"]["ticktext"])
  # Reorder heatmap
  df = df[side_dendro_leaves, :] # type: ignore
  df = df[:, upper_dendro_leaves] # type: ignore

  # heatmap
  heatmap = go.Heatmap(
    x = upper_dendro_leaves,
    y = side_dendro_leaves,
    z = df.to_numpy(),
  )

  # Under construction
  return heatmap
  