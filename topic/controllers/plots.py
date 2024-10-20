from typing import Sequence, cast
from common.ipc.requests import IPCRequestData, TopicSimilarityVisualizationMethod
import plotly.express

from common.ipc.responses import IPCResponseData
from common.ipc.task import IPCTask, TaskStepTracker
from topic.controllers.utils import assert_column_exists
from wordsmith.data.config import Config

def hierarchical_topic_plot(task: IPCTask):
  steps = TaskStepTracker(
    max_steps = 4,
  )
  message = cast(IPCRequestData.TopicPlot, task.request)

  config = Config.from_project(message.project_id)

  task.progress(0, f"Loading topic information for {message.col}.")
  model = config.paths.load_bertopic(message.col)
  
  task.progress(steps.advance(), f"Loading workspace table.")
  df = config.paths.load_workspace()

  documents = cast(list[str], assert_column_exists(df, message.col))

  task.progress(steps.advance(), f"Calculating topic hierarchy for {message.col}.")
  hierarchical_topics = model.hierarchical_topics(documents)

  task.progress(steps.advance(), f"Visualizing topic hierarchy for {message.col}.")
  sunburst = plotly.express.sunburst(
    hierarchical_topics,
    names="Topics",
    parents="Parent ID",
    values="Distance"
  )

  topic_words = cast(dict[str, Sequence[tuple[str, float]]], model.get_topics())
  
  task.success(IPCResponseData.Topics(
    plot=cast(str, sunburst.to_json()),
    topic_words=topic_words
  ), None)

def topic_correlation_plot(task: IPCTask):
  steps = TaskStepTracker(
    max_steps = 2,
  )
  message = cast(IPCRequestData.TopicCorrelationPlot, task.request)

  config = Config.from_project(message.project_id)

  task.progress(0, f"Loading topic information for {message.col}.")
  model = config.paths.load_bertopic(message.col)

  task.progress(steps.advance(), f"Calculating topic correlation for {message.col}.")

  if message.visualization == TopicSimilarityVisualizationMethod.Heatmap:
    title = f"{message.col.capitalize()} Topic Similarity Matrix"
    heatmap = model.visualize_heatmap(title=title)
    task.success(IPCResponseData.Plot(
      plot=cast(str, heatmap.to_json()),
    ), None)
  else:
    title = f"{message.col.capitalize()} Topics Distribution"
    fig = model.visualize_topics(title=title)
    task.success(IPCResponseData.Plot(
      plot=cast(str, fig.to_json()),
    ), None)
  
__all__ = [
  "topic_correlation_plot",
  "hierarchical_topic_plot",
]