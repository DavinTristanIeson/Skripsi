from dataclasses import dataclass
from modules.topic.bertopic_ext.builder import BERTopicModelBuilder
from modules.topic.experiments.model import BERTopicHyperparameterCandidate
from modules.topic.procedure.base import BERTopicProcedureComponent


class BERTopicModelBuilderProcedureComponent(BERTopicProcedureComponent):
  def run(self):
      # Dependencies
    column = self.state.column
    config = self.state.config
    documents = self.state.documents

    # Effect
    self.state.model = BERTopicModelBuilder(
      project_id=config.project_id,
      column=column,
      corpus_size=len(documents)
    ).build()


@dataclass
class BERTopicExperimentalModelBuilderProcedureComponent(BERTopicProcedureComponent):
  candidate: BERTopicHyperparameterCandidate
  def run(self):
      # Dependencies
    column = self.state.column.model_copy()
    config = self.state.config
    documents = self.state.documents

    column.topic_modeling.clustering_conservativeness = self.candidate.clustering_conservativeness
    column.topic_modeling.min_topic_size = self.candidate.min_topic_size
    column.topic_modeling.max_topics = self.candidate.max_topics
    # Effect
    self.state.model = BERTopicModelBuilder(
      project_id=config.project_id,
      column=column,
      corpus_size=len(documents)
    ).build()