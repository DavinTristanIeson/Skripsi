import json
import os
from typing import Sequence, cast

import numpy as np
import sklearn.feature_extraction
from common.ipc.requests import IPCRequestData
from common.ipc.responses import IPCResponseData
from common.ipc.task import IPCTask, TaskStepTracker

from common.utils.string import truncate_strings
from wordsmith.data.paths import ProjectPaths
import wordsmith.topic
from wordsmith.data.config import Config
from wordsmith.data.schema import TextualSchemaColumn
import plotly.express
import pandas as pd

from wordsmith.topic.evaluation import ColumnTopicsEvaluationResult, ProjectTopicsEvaluationResult
from wordsmith.topic.interpret import bertopic_vectorize

def evaluate_topics(task: IPCTask):
  steps = TaskStepTracker(
    max_steps = 5,
  )
  message = cast(IPCRequestData.Evaluation, task.request)
  config = Config.from_project(message.project_id)
  task.progress(0, f"Loading topic information for {message.project_id} > {message.column}.")
  model = config.paths.load_bertopic(message.column)



  task.progress(steps.advance(), "Loading workspace table.")
  df = config.paths.load_workspace()
  column = cast(TextualSchemaColumn, config.data_schema.assert_exists(message.column))

  texts = df[column.preprocess_column]
  texts = texts[texts != '']
  
  topic_words = wordsmith.topic.interpret.bertopic_topic_words(model)
  task.progress(steps.advance(), "Calculating C_V scores... this may take a while.")

  # We already have preprocessed min df and max df. There's no need to filter the words anymore.
  corpus, words = bertopic_vectorize(model, cast(Sequence[str], texts))

  cv_score, cv_scores_per_topic_raw = wordsmith.topic.evaluation.cv_coherence(topic_words, corpus)
  cv_scores_per_topic = np.array(cv_scores_per_topic_raw)

  task.progress(steps.advance(), "Calculating topic diversity score.")
  diversity = wordsmith.topic.evaluation.topic_diversity(topic_words)

  task.progress(steps.advance(), "Plotting C_V per topic scores.")
  topics = wordsmith.topic.interpret.bertopic_topic_labels(model)

  cv_df = pd.DataFrame({
    "Topic": tuple(truncate_strings(topics)),
    "Score": cv_scores_per_topic[:, 0],
    "Standard Deviation": cv_scores_per_topic[:, 1],
  })
  
  cv_barchart = plotly.express.bar(cv_df, x="Score", y="Topic", orientation='h', error_x="Standard Deviation")

  cv_barchart.update_traces(dict(
    customdata=topics,
    hovertemplate="<br>".join([
      "Topic: %{customdata}",
      "Score: %{x}",
    ])
  ))
  cv_barchart.update_layout(
    xaxis=dict(
      title="C_V Score"
    )
  )

  evaluation_data = ColumnTopicsEvaluationResult(
    column=message.column,
    topics=topics,
    cv_score=cv_score,
    cv_topic_scores=list(cv_scores_per_topic[:, 0]),
    topic_diversity_score=diversity,
    cv_barchart=cast(str, cv_barchart.to_json()),
  )
  evaluation_data_path = config.paths.full_path(os.path.join(ProjectPaths.Evaluation))
  if os.path.exists(evaluation_data_path):
    project_evaluation = config.paths.load_evaluation(message.column)
  else:
    project_evaluation = ProjectTopicsEvaluationResult.model_validate(dict())
  project_evaluation.root[message.column] = evaluation_data

  with open(evaluation_data_path, 'w') as f:
    json.dump(project_evaluation.model_dump(), f, indent=4)

  task.success(IPCResponseData.Evaluation(
    cv_barchart=cast(str, cv_barchart.to_json()),
    column=message.column,
    topics=topics,
    cv_score=cv_score,
    cv_topic_scores=list(cv_scores_per_topic[:, 0]),
    topic_diversity_score=diversity,
  ), f"The topics of {message.column} has been successfully evaluated. Check out the quality of the topics discovered by the topic modeling algorithm with these scores; even though they may be harder to interpret than classification scores like accuracy or precision.")