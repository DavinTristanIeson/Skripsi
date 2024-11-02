from typing import Sequence, cast
from gensim.models.coherencemodel import CoherenceModel
from common.ipc.requests import IPCRequestData
from common.ipc.responses import IPCResponseData
from common.ipc.task import IPCTask, TaskStepTracker
from wordsmith.data.config import Config
from wordsmith.data.schema import TextualSchemaColumn
import plotly.express
import pandas as pd

from wordsmith.topic.interpret import bertopic_topic_labels

def topic_diversity(topics: Sequence[Sequence[str]]):
  # based on OCTIS implementation and the equation in https://www.researchgate.net/publication/343173999_Topic_Modeling_in_Embedding_Spaces
  # https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/diversity_metrics.py#L12
  total_words = 0
  unique_words: set[str] = set()
  for topic in topics:
    unique_words |= set(topic)
    total_words += len(topic)
  td = (1 - (len(unique_words) / total_words)) ** 2
  return td

def evaluate_topics(task: IPCTask):
  steps = TaskStepTracker(
    max_steps = 5,
  )
  config = Config.from_project(task.id)
  message = cast(IPCRequestData.Evaluation, task.request)
  task.progress(0, f"Loading topic information for {message.column}.")
  model = config.paths.load_bertopic(message.column)

  task.progress(steps.advance(), "Loading workspace table.")
  df = config.paths.load_workspace()
  column = cast(TextualSchemaColumn, config.data_schema.assert_exists(message.column))

  texts = df[column.preprocess_column]
  texts = texts[texts != '']

  corpus = tuple(map(lambda doc: doc.split(), texts))
  # Error in BERTopic typing
  topic_probabilities = cast(
    Sequence[Sequence[tuple[str, float]]],
    tuple(model.get_topics().values())
    [model._outliers:]
  )
  topic_words = tuple(map(
    lambda distribution: tuple(map(
      lambda el: el[0],
      distribution
    )),
    topic_probabilities
  ))

  print(topic_probabilities, topic_words)
  task.progress(steps.advance(), "Calculating C_V scores... this may take a while.")

  cv_coherence = CoherenceModel(
    topics=topic_words,
    corpus=corpus,
    coherence='c_v'
  )
  umass_coherence = CoherenceModel(
    topics=topic_words,
    corpus=corpus,
    coherence='u_mass'
  )
  cv_score = cv_coherence.get_coherence()
  cv_scores_per_topic = cv_coherence.get_coherence_per_topic(
    with_std=True,
    with_support=True,
  )
  print(cv_score, cv_scores_per_topic)

  task.progress(steps.advance(), "Calculating topic diversity score.")
  diversity = topic_diversity(topic_words)

  task.progress(steps.advance(), "Plotting C_V per topic scores.")
  topics = bertopic_topic_labels(model, outliers=False)

  cv_df = pd.DataFrame([cv_scores_per_topic, topics], columns=["Score"])

  print(cv_df)
  
  cv_barchart = plotly.express.bar(cv_df)
  cv_barchart.update_layout(
    xaxis=dict(
      title="C_V Score"
    )
  )

  task.success(IPCResponseData.Evaluation(
    column=message.column,
    topics=topics,
    cv_score=cv_score,
    cv_barchart=cast(str, cv_barchart.to_json()),
    cv_topic_scores=cv_scores_per_topic,
    topic_diversity_score=diversity,
  ), f"The topics of {message.column} has been successfully evaluated. Check out the quality of the topics discovered by the topic modeling algorithm with these scores; even though they may be harder to interpret than classification scores like accuracy or precision.")