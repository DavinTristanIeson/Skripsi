import os
from typing import Optional

import numpy as np
from modules.logger.provisioner import ProvisionedLogger
from modules.logger.time import TimeLogger
from modules.project.cache_manager import ProjectCacheManager
from modules.project.paths import ProjectPaths
from modules.topic.procedure.base import BERTopicProcedureComponent


logger = ProvisionedLogger().provision("Topic Modeling")

class BERTopicTopicModelingProcedureComponent(BERTopicProcedureComponent):
  def run(self):
    from bertopic import BERTopic

    # Dependencies
    column = self.state.column
    config = self.state.config
    documents = list(self.state.documents) # pd.Series has issues with BERTopic code
    document_vectors = self.state.document_vectors
    model = self.state.model


    bertopic_path = config.paths.full_path(os.path.join(ProjectPaths.BERTopic(column.name)))

    if os.path.exists(bertopic_path):
      # Cache
      self.task.log_pending(f"Loading cached BERTopic model for \"{column.name}\" from \"{bertopic_path}\".")
      new_model: Optional[BERTopic] = None
      try:
        loaded_model = BERTopic.load(bertopic_path, embedding_model=model.embedding_model)
        loaded_model.umap_model = model.umap_model
        loaded_model.hdbscan_model = model.hdbscan_model

        new_model = loaded_model
        self.task.log_success(f"Loaded cached BERTopic model for \"{column.name}\" from \"{bertopic_path}\".")
      except Exception as e:
        self.task.log_error(f"Failed to load cached BERTopic model from {bertopic_path}. Re-fitting BERTopic model again.")
        logger.error(e)

      if new_model:
        # Only runs if we successfully loaded cached BERTopic
        topics_of_model_is_synced_with_current_documents = new_model.topics_ and len(new_model.topics_) == len(documents)
        if topics_of_model_is_synced_with_current_documents:
          self.state.model = new_model
          self.state.document_topic_assignments = np.array(new_model.topics_, dtype=np.int32)
          return
        # Can't use cached model.
        self.task.log_error(f"Cached BERTopic model in {bertopic_path} is not synchronized with current dataset. Re-fitting BERTopic model again.")

    self.task.log_pending(f"Starting the topic modeling process for \"{column.name}\".")

    self.task.logger.debug("Input shapes", len(documents), document_vectors.shape)

    with TimeLogger("Topic Modeling", "Performing Topic Modeling", report_start=True):
      topics, probs = model.fit_transform(documents, document_vectors)

    self.task.log_success(f"Finished the topic modeling process for {column.name}. Performing additional post-processing for the discovered topics.")
    logger.info(f"Topics of {column.name}: {model.topic_labels_}. ")

    if column.topic_modeling.no_outliers:
      topics = model.reduce_outliers(documents, topics, strategy="embeddings", embeddings=document_vectors)

    # Effect
    self.state.model = model
    self.state.document_topic_assignments = np.array(topics, dtype=np.int32)

    self.task.log_pending(f"Saving... BERTopic model in \"{bertopic_path}\".")
    cache = ProjectCacheManager().get(project_id=config.project_id)

    cache.bertopic_models.save(model, column.name)
    self.task.log_success(f"Saved BERTopic model in \"{bertopic_path}\".")

class BERTopicExperimentalTopicModelingProcedureComponent(BERTopicProcedureComponent):
  def run(self):
    # Dependencies
    column = self.state.column
    documents = list(self.state.documents) # pd.Series has issues with BERTopic code
    document_vectors = self.state.document_vectors
    model = self.state.model
    self.task.log_pending(f"Starting the topic modeling process for \"{column.name}\".")

    with TimeLogger("Topic Modeling", "Performing Topic Modeling", report_start=True):
      topics, probs = model.fit_transform(documents, document_vectors)

    self.task.log_success(f"Finished the topic modeling process for {column.name}. Performing additional post-processing for the discovered topics.")
    logger.info(f"Topics of {column.name}: {model.topic_labels_}. ")

    if column.topic_modeling.no_outliers:
      topics = model.reduce_outliers(documents, topics, strategy="embeddings", embeddings=document_vectors)

    # Effect
    self.state.model = model
    self.state.document_topic_assignments = np.array(topics, dtype=np.int32)


class BERTopicCacheOnlyTopicModelingProcedureComponent(BERTopicProcedureComponent):
  def run(self):
    from bertopic import BERTopic

    # Dependencies
    column = self.state.column
    config = self.state.config
    model = self.state.model

    # Cache
    bertopic_path = config.paths.full_path(os.path.join(ProjectPaths.BERTopic(column.name)))
    self.task.log_pending(f"Loading cached BERTopic model for \"{column.name}\" from \"{bertopic_path}\".")
    try:
      model = BERTopic.load(bertopic_path, embedding_model=model.embedding_model)
      model.umap_model = model.umap_model
      model.hdbscan_model = model.hdbscan_model
      self.state.model = model
      self.task.log_success(f"Loaded cached BERTopic model for \"{column.name}\" from \"{bertopic_path}\".")
    except Exception as e:
      self.task.log_error(f"Failed to load cached BERTopic model from {bertopic_path}. Re-fitting BERTopic model again.")
      logger.error(e)

    # Effect
    self.state.model = model
    self.state.document_topic_assignments = np.array(model.topics_, dtype=np.int32)

__all__ = [
  "BERTopicTopicModelingProcedureComponent"
]