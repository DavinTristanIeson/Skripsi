import functools
import http
import itertools
from typing import TYPE_CHECKING, Sequence, cast
import numpy as np
import pandas as pd
import pydantic

from common.models.api import ApiError
from controllers.topic.builder import BERTopicIndividualModels
from controllers.topic.interpret import BERTopicCTFIDFRepresentationResult, BERTopicInterpreter, bertopic_count_topics, bertopic_extract_topics, bertopic_extract_topic_embeddings
from controllers.topic.utils import BERTopicColumnIntermediateResult
from models.topic.topic import TopicHierarchyModel, TopicModelingResultModel, TopicModel

from .dimensionality_reduction import VisualizationCachedUMAP, VisualizationCachedUMAPResult
if TYPE_CHECKING:
  from bertopic import BERTopic


def bertopic_visualization_embeddings(
  intermediate: BERTopicColumnIntermediateResult
)->VisualizationCachedUMAPResult:
  task = intermediate.task
  config = intermediate.config
  column = intermediate.column
  model = intermediate.model
  documents = intermediate.documents
  embeddings = intermediate.embeddings    

  task.log_pending("Mapping the document and topic vectors to 2D for visualization purposes...")
  vis_umap_model = VisualizationCachedUMAP(
    project_id=config.project_id,
    column=column,
    corpus_size=len(documents),
    topic_count=bertopic_count_topics(intermediate.model)
  )
  cached_visualization_embeddings = vis_umap_model.load_cached_embeddings()
  if cached_visualization_embeddings is not None:
    return vis_umap_model.separate_embeddings(cached_visualization_embeddings)

  topic_embeddings = bertopic_extract_topic_embeddings(model)
  high_dimensional_embeddings = np.vstack([embeddings, topic_embeddings])
  visualization_embeddings = vis_umap_model.fit_transform(high_dimensional_embeddings)
  task.log_success(f"Finished mapping the document and topic vectors to 2D. The embeddings have been stored in {vis_umap_model.embedding_path}.")
  return vis_umap_model.separate_embeddings(visualization_embeddings)

class __HierarchicalClusteringGraphNodeData(pydantic.BaseModel):
  diff: float
  is_base: bool

def bertopic_hierarchical_clustering(intermediate: BERTopicColumnIntermediateResult)->TopicHierarchyModel:
  import sklearn.metrics
  import scipy.spatial.distance
  import scipy.cluster.hierarchy
  import scipy.sparse
  import networkx as nx
  
  model = intermediate.model
  task = intermediate.task
  documents = intermediate.documents
  document_topic_assignments = intermediate.document_topic_assignments
  column = intermediate.column
  
  task.log_pending(f"Performing hierarchical clustering on the topics of \"{intermediate.column.name}\" to create a topic hierarchy...")

  # Classic hierarchical clustering
  topic_embeddings: np.ndarray = bertopic_extract_topic_embeddings(model)
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
    G.add_node(
      node.id,
      diff=node.dist,
      is_base=node.left is None and node.right is None
    )
    if node.left is not None:
      G.add_edge(node.id, node.left.id)
      explore_hierarchy(G, node.left)
    if node.right is not None:
      G.add_edge(node.id, node.right.id)
      explore_hierarchy(G, node.right)

  # Simplified hierarchical clustering
  def simplify_dendrogram(G: nx.DiGraph, node: int, threshold: float):
    raw_ndata = G.nodes.get(node)
    ndata = __HierarchicalClusteringGraphNodeData.model_validate(raw_ndata)
    successors = list(G.successors(node))

    if ndata is None or ndata.diff == 0:
      return
    
    for successor in successors:
      simplify_dendrogram(G, successor, threshold)

    if ndata.diff <= threshold:
      updated_successors = list(G.successors(node))
      predecessors = list(G.predecessors(node))
      if len(predecessors) != 0:
        parent = predecessors[0]
        for successor in updated_successors:
          G.add_edge(parent, successor)
        G.remove_node(node)
    

  explore_hierarchy(
    G=dendrogram,
    node=hierarchy_tree
  )
  simplified_dendrogram = cast(nx.DiGraph, dendrogram.copy())
  simplify_dendrogram(
    G=simplified_dendrogram,
    node=hierarchy_root,
    threshold=max(0.0, min(1.0, 1-intermediate.column.topic_modeling.super_topic_similarity)),
  )

  # Update node IDs with new hierarcchy
  node_relabel_mapping = dict()
  new_node_label = bertopic_count_topics(model)
  for node, raw_ndata in sorted(simplified_dendrogram.nodes(data=True)):
    ndata = __HierarchicalClusteringGraphNodeData.model_validate(raw_ndata)
    if ndata.is_base:
      # DO NOT RENAME BASE TOPICS
      continue
    node_relabel_mapping[node] = new_node_label
    new_node_label += 1
  nx.relabel_nodes(simplified_dendrogram, node_relabel_mapping, copy=False)
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
  
  for node, raw_ndata in simplified_dendrogram.nodes(data=True):
    ndata = __HierarchicalClusteringGraphNodeData.model_validate(raw_ndata)
    if not ndata.is_base:
      continue
    documents_in_this_node = cast(Sequence[str], documents[document_topic_assignments == node])
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
      children.append(build_topic_hierarchy(G, successor))
    try:
      frequency = cast(int, model.get_topic_freq(node))
    except KeyError:
      frequency = functools.reduce(lambda acc, cur: acc + cur.frequency, children, 0)

    label = ', '.join(itertools.islice(map(
      lambda x: x[0],
      filter(
        lambda x: len(x[0]) > 0 and x[1] > 0,
        representation_results[node].words
      )
    ), 3))

    return TopicHierarchyModel(
      id=node,
      frequency=frequency,
      label=label,
      words=representation_results[node].words,
      children=children,
    )
  
  topic_hierarchy = build_topic_hierarchy(simplified_dendrogram, hierarchy_root)

  task.log_success(f"Finished performing hierarchical clustering on the topics of \"{intermediate.column.name}\".")
  return topic_hierarchy


def bertopic_post_processing(df: pd.DataFrame, intermediate: BERTopicColumnIntermediateResult)->TopicModelingResultModel:
  column = intermediate.column
  model = intermediate.model
  task = intermediate.task

  task.log_pending(f"Applying post-processing on the topics of \"{intermediate.column.name}\"...")

  # Set topic assignments
  document_topic_mapping_column = pd.Series(np.full(len(df), -1), dtype=np.int32)
  document_topic_mapping_column[intermediate.mask] = model.topics_
  document_topic_mapping_column[~intermediate.mask] = pd.NA
  df[column.topic_column.name] = document_topic_mapping_column

  # Set topic labels
  topics = bertopic_extract_topics(model)

  # Perform hierarchical clustering
  hierarchy = bertopic_hierarchical_clustering(intermediate)

  # Embed document/topics
  bertopic_visualization_embeddings(intermediate).topic_embeddings

  # Create topic result
  topic_modeling_result = TopicModelingResultModel(
    project_id=intermediate.config.project_id,
    topics=topics,
    hierarchy=hierarchy,
    frequency=len(intermediate.documents),
  )

  task.log_success(f"Finished appling post-processing on the topics of \"{intermediate.column.name}\".")
  return topic_modeling_result

  