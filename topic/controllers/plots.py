from typing import Sequence, cast
from common.ipc.client import IntraProcessCommunicator
from common.ipc.requests import IPCRequestData, TopicSimilarityVisualizationMethod
import plotly.express

from common.ipc.responses import IPCResponseData
from common.ipc.tasks import ipc_task_handler
from common.models.api import ApiError
from topic.controllers.utils import assert_column_exists
from wordsmith.data.config import Config

@ipc_task_handler
def hierarchical_topic_plot(comm: IntraProcessCommunicator, message: IPCRequestData.TopicPlot):
  config = Config.from_project(message.project_id)
  model = config.paths.load_bertopic(message.col)
  
  df = config.paths.load_workspace()
  documents = cast(list[str], assert_column_exists(df, message.col))
  hierarchical_topics = model.hierarchical_topics(documents)

  sunburst = plotly.express.sunburst(
    hierarchical_topics,
    names="Topics",
    parents="Parent ID",
    values="Distance"
  )

  topic_words = cast(dict[str, Sequence[tuple[str, float]]], model.get_topics())
  return IPCResponseData.TopicPlot(
    plot=cast(str, sunburst.to_json()),
    topic_words=topic_words
  )

@ipc_task_handler
def topic_correlation_plot(comm: IntraProcessCommunicator, message: IPCRequestData.TopicCorrelationPlot):
  config = Config.from_project(message.project_id)
  model = config.paths.load_bertopic(message.col)
  if message.visualization == TopicSimilarityVisualizationMethod.Heatmap:
    title = f"{message.col.capitalize()} Topic Similarity Matrix"
    heatmap = model.visualize_heatmap(title=title)
    return IPCResponseData.Plot(
      plot=cast(str, heatmap.to_json()),
    )
  else:
    title = f"{message.col.capitalize()} Topics Distribution"
    fig = model.visualize_topics(title=title)
    return IPCResponseData.Plot(
      plot=cast(str, fig.to_json()),
    )
  