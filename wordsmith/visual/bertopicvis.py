# Transforms the hierarchical topics computed by BERTopic .hierarchical_topics to a sunburst plot
import pandas as pd
import plotly.express


def hierarchical_topics_sunburst(hierarchical_topics: pd.DataFrame, topic_labels: dict[int, str], topic_freq: dict[int, int]):
  parents: dict[int, int] = {}
  labels: dict[int, str] = {**topic_labels}
  freqs: dict[int, int] = {**topic_freq}
  
  new_parent = len(topic_freq) + 1
  for idx, row in hierarchical_topics.iterrows():
    topics = row["Topics"]
    parent_name = row["Parent_Name"]

    total_freqs = 0
    for child in topics:
      if child not in parents:
        # Give parent to orphan
        parents[child] = new_parent
      else:
        grandparent = parents.get(new_parent, -1)
        # Get the latest ancestor if a grandfather conflict occurs
        parents[new_parent] = max(grandparent, parents[child])
        parents[child] = new_parent
      total_freqs += topic_freq[child]

    labels[new_parent] = parent_name
    freqs[new_parent] = total_freqs
    new_parent += 1

  df_rows = []
  for child, parent in parents.items():
    df_rows.append([
      child,
      parent,
      labels[child],
      freqs[child],
    ])

  df = pd.DataFrame(df_rows, columns=["ID", "Parent", "Label", "Frequency"])
  total_documents = sum(topic_freq.values())
  df["Percentage"] = ((df["Frequency"] / total_documents) * 100).map(lambda x: f"{x:.3f}")

  fig = plotly.express.sunburst(
    df,
    ids="ID",
    names="Label",
    parents="Parent",
    values="Frequency",
    hover_data=df.loc[:, ["Label", "Frequency", "Percentage"]]
  )

  fig.update_traces(
    hovertemplate="<br>".join([
      "Topic: %{customdata[0]}",
      "Frequency: %{customdata[1]} (%{customdata[2]}%)",
    ])
  )

  return fig