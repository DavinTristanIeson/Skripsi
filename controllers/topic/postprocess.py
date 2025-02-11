import functools
import http
from typing import TYPE_CHECKING, Sequence, cast
import numpy as np
import pandas as pd

from common.models.api import ApiError
from controllers.topic.builder import BERTopicIndividualModels
from controllers.topic.interpret import BERTopicCTFIDFRepresentationResult, BERTopicInterpreter, bertopic_count_topics, bertopic_topic_words
from controllers.topic.utils import BERTopicColumnIntermediateResult
from models.topic.topic import TopicHierarchyModel

from .dimensionality_reduction import VisualizationCachedUMAP
if TYPE_CHECKING:
  from bertopic import BERTopic


def bertopic_visualization_embeddings(
  intermediate: BERTopicColumnIntermediateResult
):
  task = intermediate.task
  config = intermediate.config
  column = intermediate.column
  documents = intermediate.documents
  cached_umap_model = BERTopicIndividualModels.cast(intermediate.model).umap_model
  reduced_embeddings = cached_umap_model.load_cached_embeddings()
  if reduced_embeddings is None:
    task.error(ApiError(f"Unable to find the embeddings created by UMAP in \"{cached_umap_model.embedding_path}\". The topic modeling procedure might not have been executed yet. Please re-run the topic modeling procedure to fix this.", http.HTTPStatus.INTERNAL_SERVER_ERROR))
    return
  task.progress("Mapping the document and topic vectors to 2D for visualization purposes...")
  topic_embeddings = intermediate.model.topic_embeddings_
  high_dimensional_embeddings = np.vstack([reduced_embeddings, topic_embeddings]) # type: ignore
  umap_model = VisualizationCachedUMAP(
    paths=config.paths,
    column=column,
    corpus_size=len(documents),
    topic_count=bertopic_count_topics(intermediate.model)
  )
  umap_model.fit_transform(high_dimensional_embeddings)
  task.progress(f"Finished mapping the document and topic vectors to 2D. The embeddings have been stored in {umap_model.embedding_path}.")

def bertopic_hierarchical_clustering(intermediate: BERTopicColumnIntermediateResult):
  import sklearn.metrics
  import scipy.spatial.distance
  import scipy.cluster.hierarchy
  import scipy.sparse
  import networkx as nx
  
  model = intermediate.model
  task = intermediate.task
  documents = intermediate.documents
  topics = intermediate.document_topic_assignments
  column = intermediate.column
  
  task.progress(f"Performing hierarchical clustering on the topics of \"{intermediate.column.name}\" to create a topic hierarchy...")

  # Classic hierarchical clustering
  topic_embeddings: np.ndarray = cast(np.ndarray, model.topic_embeddings_)
  intertopic_distances = sklearn.metrics.pairwise.cosine_distances(topic_embeddings)
  condensed_intertopic_distances = scipy.spatial.distance.squareform(intertopic_distances)

  hierarchy_tree: scipy.cluster.hierarchy.ClusterNode = cast(
    scipy.cluster.hierarchy.ClusterNode,
    scipy.cluster.hierarchy.to_tree(
      scipy.cluster.hierarchy.ward(condensed_intertopic_distances)
    )
  )
  hierarchy_root = hierarchy_tree.id
  dendrogram = nx.DiGraph()

  def explore_hierarchy(G: nx.DiGraph, node: scipy.cluster.hierarchy.ClusterNode):
    G.add_node(node.id, diff=node.dist)
    if node.left is not None:
      G.add_edge(node.id, node.left.id)
      explore_hierarchy(G, node.left)
    if node.right is not None:
      G.add_edge(node.id, node.right.id)
      explore_hierarchy(G, node.right)

  # Simplified hierarchical clustering
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
    node=hierarchy_root,
    threshold=1 + intermediate.column.topic_modeling.super_topic_similarity
  )

  # Update node IDs with new hiearchy
  node_relabel_mapping = dict()
  new_node_label = topic_embeddings.shape[0]
  for node in sorted(simplified_dendrogram.nodes):
    node_relabel_mapping[node] = new_node_label
    new_node_label += 1
  nx.relabel_nodes(simplified_dendrogram, node_relabel_mapping)
  hierarchy_root = node_relabel_mapping[hierarchy_root]

  # Initialize base BOWs
  interpreter = BERTopicInterpreter(
    topic_ctfidf=model.c_tf_idf_, # type: ignore
    ctfidf_model=model.ctfidf_model, # type: ignore
    top_n_words=column.topic_modeling.top_n_words,
    vectorizer_model=model.vectorizer_model,
  )
  documents = intermediate.documents
  representation_results: dict[int, BERTopicCTFIDFRepresentationResult] = dict()
  
  for node in simplified_dendrogram.nodes:
    documents_in_this_node = cast(Sequence[str], documents[topics == node])
    meta_document_bow = interpreter.represent_as_bow(documents_in_this_node)
    meta_document_ctfidf = interpreter.represent_as_ctfidf(meta_document_bow)
    representation = BERTopicCTFIDFRepresentationResult(
      bow=meta_document_bow,
      ctfidf=meta_document_ctfidf,
      words=interpreter.get_weighted_words(meta_document_ctfidf)
    )
    representation_results[node] = representation

  # Agglomeratively sum up BOWs to calculate topic words for each super-topic
  def label_dendrogram(G: nx.DiGraph, node: int):
    if node in representation_results:
      return representation_results[node].bow
    
    children_bows = list(map(lambda x: label_dendrogram(G, x), G.successors(node)))

    agglomerated_bow = functools.reduce(lambda acc, cur: acc + cur, children_bows[1:], children_bows[0])
    agglomerated_ctfidf = interpreter.represent_as_ctfidf(agglomerated_bow)
    agglomerated_words = interpreter.get_weighted_words(agglomerated_ctfidf)

    representation_results[node] = BERTopicCTFIDFRepresentationResult(
      ctfidf=agglomerated_ctfidf,
      bow=agglomerated_bow,
      words=agglomerated_words,
    )
    return agglomerated_bow
    
  label_dendrogram(simplified_dendrogram, hierarchy_root)

  def build_topic_hierarchy(G: nx.DiGraph, node: int):
    children: list[TopicHierarchyModel] = []
    for successor in G.successors(node):
      children.append(successor)
    return TopicHierarchyModel(
      id=node,
      words=representation_results[node].words,
      children=children,
    )
  
  topic_hierarchy = build_topic_hierarchy(simplified_dendrogram, hierarchy_root)
  return topic_hierarchy


def bertopic_post_processing(df: pd.DataFrame, intermediate: BERTopicColumnIntermediateResult):
  column = intermediate.column
  model = intermediate.model
  task = intermediate.task

  # Set topic labels
  interpreter = BERTopicInterpreter.from_model(model, documents=intermediate.documents) # type: ignore
  topic_labels = bertopic_topic_words(model).labels
  if model._outliers:
    topic_labels.insert(0, '')
  model.set_topic_labels(topic_labels)

  # Set topic assignments
  document_topic_mapping_column = pd.Series(np.full(len(intermediate.embedding_documents), -1), dtype=np.int32)
  document_topic_mapping_column[intermediate.mask] = model.topics_
  df[column.topic_column.name] = document_topic_mapping_column

  # Perform hierarchical clustering
  hierarchy = bertopic_hierarchical_clustering(intermediate)

  # Embed document/topics
  task.progress(f"Performing hierarchical clustering on the topics of \"{intermediate.column.name}\" to create a topic hierarchy...")
  bertopic_visualization_embeddings(intermediate)
