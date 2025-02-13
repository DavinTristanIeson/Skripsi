from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, cast
from common.logger import RegisteredLogger
from controllers.topic.dimensionality_reduction import BERTopicCachedUMAP
from controllers.topic.embedding import BERTopicEmbeddingModelFactory, SupportedBERTopicEmbeddingModels
from models.config.paths import ProjectPathManager, ProjectPaths
from models.config.schema import TextualSchemaColumn
if TYPE_CHECKING:
  from bertopic import BERTopic
  from bertopic.vectorizers import ClassTfidfTransformer
  from bertopic.representation import MaximalMarginalRelevance
  from hdbscan import HDBSCAN
  from umap import UMAP
  from sklearn.feature_extraction.text import CountVectorizer


logger = RegisteredLogger().provision("Topic Modeling")
@dataclass
class BERTopicIndividualModels:
  embedding_model: SupportedBERTopicEmbeddingModels
  umap_model: BERTopicCachedUMAP
  hdbscan_model: "HDBSCAN"
  vectorizer_model: "CountVectorizer"
  ctfidf_model: "ClassTfidfTransformer"
  representation_model: "MaximalMarginalRelevance"

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
class BERTopicModelBuilder:
  project_id: str
  column: TextualSchemaColumn
  corpus_size: int

  def build_embedding_model(self)->"SupportedBERTopicEmbeddingModels":
    paths = ProjectPathManager(project_id=self.project_id)
    column = self.column
    return BERTopicEmbeddingModelFactory(
      paths=paths,
      column=column
    ).build()

  def build_umap_model(self)->"UMAP":
    paths = ProjectPathManager(project_id=self.project_id)
    return BERTopicCachedUMAP(
      paths=paths,
      column=self.column,
    ) # type: ignore
  
  def build_hdbscan_model(self)->"HDBSCAN":
    from hdbscan import HDBSCAN
    column = self.column
    max_cluster_size = int(column.topic_modeling.max_topic_size * self.corpus_size)

    if column.topic_modeling.min_topic_size >= max_cluster_size:
      raise ValueError("Min. topic size should not be greater than max. topic size. Please set a higher max topic size. Note: This can also happen if you have too few valid documents to analyze.")
    
    min_cluster_size = max(2, column.topic_modeling.min_topic_size)
    min_samples = max(2, int(column.topic_modeling.clustering_conservativeness * column.topic_modeling.min_topic_size))

    hdbscan_model = HDBSCAN(
      min_cluster_size=min_cluster_size,
      max_cluster_size=max_cluster_size,
      min_samples=min_samples,
      metric="euclidean",
      cluster_selection_method="eom", # following BERTopic arguments
    )
    return hdbscan_model
  
  def build_vectorizer_model(self)->"CountVectorizer":
    from sklearn.feature_extraction.text import CountVectorizer
    column = self.column
    vectorizer_model = CountVectorizer(
      min_df=column.preprocessing.min_df,
      max_df=column.preprocessing.max_df,
      stop_words=None,
      ngram_range=column.topic_modeling.n_gram_range
    )
    return vectorizer_model
  
  def build_ctfidf_model(self)->"ClassTfidfTransformer":
    from bertopic.vectorizers import ClassTfidfTransformer
    ctfidf_model = ClassTfidfTransformer(
      bm25_weighting=True,
      reduce_frequent_words=True,
    )
    return ctfidf_model

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