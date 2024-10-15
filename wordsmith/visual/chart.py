import pandas as pd
import plotly.graph_objects as go

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
