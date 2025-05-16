from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Optional, cast

from modules.logger import ProvisionedLogger
from modules.config import TextualSchemaColumn
from modules.project.paths import ProjectPathManager, ProjectPaths

from .dimensionality_reduction import BERTopicCachedUMAP
from .embedding import BERTopicEmbeddingModelFactory, SupportedBERTopicEmbeddingModels

if TYPE_CHECKING:
  from bertopic import BERTopic
  from bertopic.vectorizers import ClassTfidfTransformer
  from bertopic.representation import KeyBERTInspired
  from hdbscan import HDBSCAN
  from umap import UMAP
  from sklearn.feature_extraction.text import CountVectorizer



logger = ProvisionedLogger().provision("Topic Modeling")
@dataclass
class BERTopicIndividualModels:
  embedding_model: SupportedBERTopicEmbeddingModels
  umap_model: BERTopicCachedUMAP
  hdbscan_model: "HDBSCAN"
  vectorizer_model: "CountVectorizer"
  ctfidf_model: "ClassTfidfTransformer"
  representation_model: "KeyBERTInspired"

  @staticmethod
  def cast(model: "BERTopic")->"BERTopicIndividualModels":
    return BERTopicIndividualModels(
      embedding_model=model.embedding_model, # type: ignore
      umap_model=model.umap_model, # type: ignore
      hdbscan_model=model.hdbscan_model, # type: ignore
      vectorizer_model=model.vectorizer_model, # type: ignore
      ctfidf_model=model.ctfidf_model, # type: ignore
      representation_model=model.representation_model # type: ignore
    )

@dataclass
class EmptyBERTopicModelBuilder:
  column: TextualSchemaColumn
  def build_vectorizer_model(self)->"CountVectorizer":
    from sklearn.feature_extraction.text import CountVectorizer
    # We handle all preprocessing.
    vectorizer_model = CountVectorizer(
      min_df=1,
      max_df=1.0,
      stop_words=None,
    )
    return vectorizer_model
  
  def build_ctfidf_model(self)->"ClassTfidfTransformer":
    from bertopic.vectorizers import ClassTfidfTransformer
    ctfidf_model = ClassTfidfTransformer(
      bm25_weighting=True,
      reduce_frequent_words=True,
    )
    return ctfidf_model
  
  def build(self):
    from bertopic import BERTopic
    from bertopic.backend import BaseEmbedder
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.cluster import BaseCluster
    return BERTopic(
      embedding_model=BaseEmbedder(),
      umap_model=BaseDimensionalityReduction(),
      hdbscan_model=BaseCluster(),
      vectorizer_model=self.build_vectorizer_model(),
      ctfidf_model=self.build_ctfidf_model(),
      top_n_words=self.column.topic_modeling.top_n_words,
    )

@dataclass
class BERTopicModelBuilder:
  project_id: str
  column: TextualSchemaColumn
  corpus_size: Optional[int]

  def build_embedding_model(self)->"SupportedBERTopicEmbeddingModels":
    column = self.column
    return BERTopicEmbeddingModelFactory(
      project_id=self.project_id,
      column=column
    ).build()

  def build_umap_model(self)->"UMAP":
    from umap import UMAP
    return cast(UMAP, BERTopicCachedUMAP(
      project_id=self.project_id,
      column=self.column,
      low_memory=True,
    ))
  
  def build_hdbscan_model(self)->"HDBSCAN":
    from hdbscan import HDBSCAN
    column = self.column
    max_cluster_size = self.corpus_size and column.topic_modeling.max_topic_size and int(column.topic_modeling.max_topic_size * self.corpus_size)
    min_cluster_size = column.topic_modeling.min_topic_size
    min_samples = column.topic_modeling.topic_confidence_threshold

    params = dict()
    if max_cluster_size is not None:
      if min_cluster_size >= max_cluster_size:
        raise ValueError("Min. topic size should not be greater than max. topic size. Please set a higher max topic size. Note: This can also happen if you have too few valid documents to analyze.")
      
      params["max_cluster_size"] = max_cluster_size


    hdbscan_model = HDBSCAN(
      min_cluster_size=min_cluster_size,
      min_samples=min_samples,
      metric="euclidean",
      cluster_selection_method="eom", # following BERTopic arguments
      **params,
    )
    return hdbscan_model
  
  def build_vectorizer_model(self)->"CountVectorizer":
    return EmptyBERTopicModelBuilder(self.column).build_vectorizer_model()
  
  def build_ctfidf_model(self)->"ClassTfidfTransformer":
    return EmptyBERTopicModelBuilder(self.column).build_ctfidf_model()
  
  def build_representation_model(self)->"KeyBERTInspired":
    from bertopic.representation import KeyBERTInspired
    return KeyBERTInspired(
      top_n_words=self.column.topic_modeling.top_n_words
    )

  def build(self)->"BERTopic":
    from bertopic import BERTopic

    column = self.column

    kwargs = dict()
    if column.topic_modeling.max_topics is not None:
      kwargs["nr_topics"] = column.topic_modeling.max_topics

    model = BERTopic(
      embedding_model=self.build_embedding_model(),
      umap_model=self.build_umap_model(),
      hdbscan_model=self.build_hdbscan_model(),
      ctfidf_model=self.build_ctfidf_model(),
      vectorizer_model=self.build_vectorizer_model(),
      representation_model=self.build_representation_model(),
      top_n_words=int(column.topic_modeling.top_n_words),
      calculate_probabilities=False,
      verbose=True,
      **kwargs,
    )
    return model
    
  def load(self):
    paths = ProjectPathManager(project_id=self.project_id)
    bertopic_path = paths.full_path(os.path.join(ProjectPaths.BERTopic(self.column.name)))
    if not os.path.exists(bertopic_path):
      return None
    try:
      model = BERTopic.load(bertopic_path)
    except Exception as e:
      logger.error(e)
      return None
    
    model.embedding_model = self.build_embedding_model()
    model.umap_model = self.build_umap_model()
    model.hdbscan_model = self.build_hdbscan_model()

    return model
  
__all__ = [
  "BERTopicIndividualModels",
  "BERTopicModelBuilder",
  "EmptyBERTopicModelBuilder"
]