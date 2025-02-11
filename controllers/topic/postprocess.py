import http
from typing import TYPE_CHECKING, cast
import numpy as np
import numpy.typing as npt
import pandas as pd

from common.models.api import ApiError
from controllers.topic.interpret import bertopic_topic_labels
from controllers.topic.utils import BERTopicColumnIntermediateResult
from models.topic.topic import TopicModelingResultModel

from .dimensionality_reduction import VisualizationCachedUMAP, BERTopicCachedUMAP
if TYPE_CHECKING:
  from bertopic import BERTopic

def bertopic_visualization_embeddings(
  intermediate: BERTopicColumnIntermediateResult
):
  task = intermediate.task
  config = intermediate.config
  column = intermediate.column
  cached_umap_model = BERTopicCachedUMAP(
    column=column,
    paths=config.paths
  )
  if not cached_umap_model.has_cached_embeddings():
    task.error(ApiError(f"Unable to find the embeddings created by UMAP in \"{cached_umap_model.embedding_path}\". The topic modeling procedure might not have been executed yet. Please re-run the topic modeling procedure to fix this.", http.HTTPStatus.INTERNAL_SERVER_ERROR))
    return
  reduced_embeddings = cached_umap_model.load_cached_embeddings()
  topic_embeddings = intermediate.model.topic_embeddings_
  high_dimensional_embeddings = np.vstack([reduced_embeddings, topic_embeddings]) # type: ignore
  umap_model = VisualizationCachedUMAP(
    paths=config.paths,
    column=column,
  )
  visualization_embeddings = umap_model.fit_transform(high_dimensional_embeddings)
  low_document_embeddings = visualization_embeddings[:len(reduced_embeddings)]
  low_topic_embeddings = visualization_embeddings[len(reduced_embeddings):]

  intermediate.document_visualization_embeddings = low_document_embeddings
  intermediate.topic_visualization_embeddings = low_topic_embeddings

def bertopic_hierarchical_clustering(intermediate: BERTopicColumnIntermediateResult):
  import sklearn.metrics
  import scipy.spatial.distance
  import scipy.cluster.hierarchy
  import networkx as nx
  
  model = intermediate.model
  task = intermediate.task
  documents = intermediate.documents

  
  task.progress(f"Performing hierarchical clustering on the topics of \"{intermediate.column.name}\" to create a topic hierarchy...")

  topic_embeddings: npt.NDArray = cast(npt.NDArray, model.topic_embeddings_)
  intertopic_distances = sklearn.metrics.pairwise.cosine_distances(topic_embeddings)
  condensed_intertopic_distances = scipy.spatial.distance.squareform(intertopic_distances)

  hierarchy_tree: scipy.cluster.hierarchy.ClusterNode = cast(
    scipy.cluster.hierarchy.ClusterNode,
    scipy.cluster.hierarchy.to_tree(
      scipy.cluster.hierarchy.ward(condensed_intertopic_distances)
    )
  )
  dendrogram = nx.DiGraph()

  def explore_hierarchy(G: nx.DiGraph, node: scipy.cluster.hierarchy.ClusterNode):
    G.add_node(node.id, diff=node.dist)
    if node.left is not None:
      G.add_edge(node.id, node.left.id)
      explore_hierarchy(G, node.left)
    if node.right is not None:
      G.add_edge(node.id, node.right.id)
      explore_hierarchy(G, node.right)

  def simplify_dendrogram(G: nx.DiGraph, node: int, threshold: float):
    ndata = G.nodes.get(node)
    successors = list(G.successors(node))

    if ndata is None or ndata["diff"] == 0:
      return
    if ndata["diff"] <= threshold:
      predecessors = list(G.predecessors(node))
      if len(predecessors) != 0:
        parent = predecessors[0]
        for successor in successors:
          G.add_edge(parent, successor)
        G.remove_node(node)
    
    for successor in successors:
      simplify_dendrogram(G, successor, threshold)


  explore_hierarchy(
    G=dendrogram,
    node=hierarchy_tree
  )

  simplified_dendrogram = cast(nx.DiGraph, dendrogram.copy())
  simplify_dendrogram(
    G=simplified_dendrogram,
    node=hierarchy_tree.id,
    threshold=1 + intermediate.column.topic_modeling.super_topic_similarity
  )

  node_relabel_mapping = dict()
  new_node_label = len(topic_embeddings)
  for node in sorted(simplified_dendrogram.nodes):
    node_relabel_mapping[node] = new_node_label
    new_node_label += 1
  nx.relabel_nodes(simplified_dendrogram, node_relabel_mapping)

  def simplify_dendrogram(G: nx.DiGraph, node: int, threshold: float):
    ndata = G.nodes.get(node)
    successors = list(G.successors(node))

    if ndata is None or ndata["diff"] == 0:
      return
    if ndata["diff"] <= threshold:
      predecessors = list(G.predecessors(node))
      if len(predecessors) != 0:
        parent = predecessors[0]
        for successor in successors:
          G.add_edge(parent, successor)
        G.remove_node(node)
    
    for successor in successors:
      simplify_dendrogram(G, successor, threshold)

  intermediate.hierarchy = simplified_dendrogram


def bertopic_post_processing(df: pd.DataFrame, intermediate: BERTopicColumnIntermediateResult):
  column = intermediate.column
  model = intermediate.model
  task = intermediate.task

  # Set topic labels
  topic_labels = bertopic_topic_labels(model)
  if model._outliers:
    topic_labels.insert(0, '')
  model.set_topic_labels(topic_labels)

  # Set topic assignments
  document_topic_mapping_column = pd.Series(np.full(len(intermediate.embedding_documents), -1), dtype=np.int32)
  document_topic_mapping_column[intermediate.mask] = model.topics_
  df[column.topic_column.name] = document_topic_mapping_column

  # Perform hierarchical clustering
  bertopic_hierarchical_clustering(intermediate)

  # Embed document/topics
  task.progress(f"Performing hierarchical clustering on the topics of \"{intermediate.column.name}\" to create a topic hierarchy...")
  bertopic_visualization_embeddings(intermediate)
