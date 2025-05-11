from modules.topic.bertopic_ext.builder import BERTopicModelBuilder
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

