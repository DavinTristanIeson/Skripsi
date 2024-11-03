import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff

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
  # Modified from https://plotly.com/python/dendrogram/

  # Upper dendrogram
  upper_dendrogram = ff.create_dendrogram(df.T, orientation="bottom", labels=df.columns)
  for i in range(len(upper_dendrogram['data'])): # type: ignore
    # Plotly internals is just a dictionary as config object
    upper_dendrogram['data'][i]['yaxis'] = 'y2'  # type: ignore
  
  # Side dendrogram
  side_dendrogram = ff.create_dendrogram(df, orientation="left", labels=df.index)
  for i in range(len(side_dendrogram['data'])): # type: ignore
    side_dendrogram['data'][i]['xaxis'] = 'x2' # type: ignore
  for trace in side_dendrogram.data:
    upper_dendrogram.add_trace(trace)

  upper_dendrogram.update_traces(dict(
    hoverinfo="skip",
    hovertemplate=None
  ))

  side_dendro_leaves: list[str] = list(side_dendrogram.layout["yaxis"]["ticktext"]) # type: ignore
  upper_dendro_leaves: list[str] = list(upper_dendrogram.layout["xaxis"]["ticktext"]) # type: ignore

  # Reorder heatmap
  df = df.loc[side_dendro_leaves, upper_dendro_leaves] # type: ignore

  # heatmap
  heatmap = go.Heatmap(
    x = upper_dendrogram['layout']['xaxis']['tickvals'], # type: ignore
    y = side_dendrogram['layout']['yaxis']['tickvals'], # type: ignore
    z = df.to_numpy(),
    name='',
  )

  upper_dendrogram.add_trace(heatmap)

  shared_params = dict(
    mirror=False,
    showgrid=False,
    showline=False,
    zeroline=False,
  )

  upper_dendrogram.update_layout(dict(
    width=800,
    height=800,
    xaxis=dict(
      **shared_params,
      ticktext=df.columns,
      domain=[0, 0.85],
    ),
    xaxis2=dict(
      **shared_params,
      domain=[0.85, 1],
      ticks='',
      showticklabels=False
    ),
    yaxis=dict(
      **shared_params,
      domain=[0, 0.85],
      ticktext=df.index,
      # For some reason, this is not automatically filled in by plotly
      tickvals=side_dendrogram['layout']['yaxis']['tickvals'] # type: ignore
    ),
    yaxis2=dict(
      **shared_params,
      domain=[0.85, 1],
      ticks='',
      showticklabels=False
    ),
  ))

  return upper_dendrogram, (side_dendro_leaves, upper_dendro_leaves)
  