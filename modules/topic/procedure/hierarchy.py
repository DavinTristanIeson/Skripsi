import functools
from typing import TYPE_CHECKING, Sequence, cast

import numpy as np
import pandas as pd
from modules.topic.bertopic_ext.interpret import BERTopicInterpreter
from modules.topic.model import Topic
from modules.topic.procedure.utils import _BERTopicColumnIntermediateResult

if TYPE_CHECKING:
  import networkx as nx

def bertopic_hierarchical_clustering(intermediate: _BERTopicColumnIntermediateResult)->nx.DiGraph:
  import sklearn.metrics.pairwise
  import networkx as nx

  interpreter = BERTopicInterpreter(intermediate.model)

  groups = intermediate.embeddings
  weights = 2 - sklearn.metrics.pairwise.cosine_distances(groups)

  layers = [[[topic] for topic in range(interpreter.topic_count)]]
  while True:
    # Normalize cosine distances
    median_weights = np.median(weights)
    mask = (weights < median_weights) & np.eye(len(weights), dtype=np.bool_)
    weights[mask] = 0
    complete_graph = nx.Graph(weights)
    minimum_spanning_tree = nx.minimum_spanning_tree(complete_graph, algorithm="kruskal")
    communities = nx.community.louvain_communities(minimum_spanning_tree) # type: ignore

    layers.append(communities)
    layer_idx = len(layers) - 1
    if len(layers[layer_idx]) == len(layers[layer_idx - 1]):
      # Just consider this as root, since there are no longer any merges
      layers[layer_idx] = [functools.reduce(lambda acc, cur: acc | cur, set())]
      break

    if len(layers[layer_idx]) == 1:
      # We've reached a root.
      break

    raw_new_groups = []
    for community in communities:
      group_centroid = groups[list(community)].mean()
      raw_new_groups.append(group_centroid)
    groups = np.array(raw_new_groups)
    weights = 2 - sklearn.metrics.pairwise.cosine_distances(groups)


  hierarchy = nx.DiGraph()
  for layer_idx, layer in enumerate(layers):
    for group_idx, group in enumerate(layer):
      # Each item is associated with their layer and index.
      node_id = f"L{layer_idx}_{group_idx}"
      hierarchy.add_node(node_id, label=str(group))

  for layer_idx in range(len(layers) - 1):
    layer = layers[layer_idx]
    for group_idx, group in enumerate(layer):
      node_id = f"L{layer_idx}_{group_idx}"
      for element in group:
        # Each element in a group points to a group in the next layer
        next_node_id = f"L{layer_idx + 1}_{element}"
        hierarchy.add_edge(node_id, next_node_id)
  
  # Simplify hierarchy

  hierarchy_root = 'L0_0'
  layers = list(nx.bfs_layers(hierarchy, hierarchy_root))
  # Skip first and last layer
  layers = layers[1:len(layers)-1]
  for layer in layers:
    for node in layer:
      parent = list(hierarchy.predecessors(node))[0]
      children = list(hierarchy.neighbors(node))
      if len(children) == 1:
        only_child = children[0]
        # Connect parent to child if this node only has one child, since it means that it becomes redundant.
        hierarchy.add_edge(parent, only_child)
        hierarchy.remove_node(node)

  return hierarchy


def calculate_weighted_words_for_topic_scopes(
  topic_scopes: dict[int, list[int]],
  documents: pd.Series,
  document_topics: pd.Series | np.ndarray,
  interpreter: BERTopicInterpreter,
)->dict[int, list[tuple[str, float]]]:
  raw_base_bows = []
  for topic in interpreter.extract_topics():
    documents_in_this_node = cast(Sequence[str], documents[document_topics == topic.id]) 
    bow = interpreter.represent_as_bow(documents_in_this_node)
    raw_base_bows.append(bow)
  base_bows = np.array(raw_base_bows)
  
  weighted_words: dict[int, list[tuple[str, float]]] = dict()
  for topic, scopes in topic_scopes.items():
    supertopic_bow = base_bows[scopes].sum()
    supertopic_ctfidf = interpreter.represent_as_ctfidf(supertopic_bow)
    weighted_words[topic] = interpreter.get_weighted_words(supertopic_ctfidf)
  return weighted_words
  
  
def __get_leaf_nodes(G: nx.DiGraph, node: int)->list[int]:
  return list(filter(lambda node: len(list(G.successors(node))) == 0, nx.dfs_successors(G, node)))
  
def bertopic_topic_hierarchy(intermediate: _BERTopicColumnIntermediateResult)->Topic:
  import networkx as nx
  interpreter = BERTopicInterpreter(intermediate.model)
  new_node_label = interpreter.topic_count
  hierarchy = bertopic_hierarchical_clustering(intermediate)
  
  # First layer guaranteed to have only one element. So this element absolutely exists.
  hierarchy_root = 'L0_0'

  new_node_label = interpreter.topic_count
  node_relabel_mapping: dict[str, int] = dict()
  for node in reversed(list(nx.bfs_successors(hierarchy, hierarchy_root, depth_limit=3))):
    node_relabel_mapping[node] = new_node_label
    new_node_label += 1
  hierarchy_root = node_relabel_mapping[hierarchy_root]
  nx.relabel_nodes(hierarchy, node_relabel_mapping, copy=False)

  # Initialize base BOWs
  documents = intermediate.documents
  document_topics = intermediate.model.topics_

  merged_topic_hierarchy = cast(nx.DiGraph, hierarchy.copy())
  leaf_nodes = __get_leaf_nodes(hierarchy, hierarchy_root)
  merged_topic_hierarchy.remove_nodes_from(leaf_nodes)
  
  topic_scopes: dict[int, list[int]] = {}
  for node in merged_topic_hierarchy.nodes:
    topic_scopes[node] = __get_leaf_nodes(merged_topic_hierarchy, node)

  base_topics = interpreter.extract_topics()
  merged_topics: list[Topic] = []

  merged_topic_weighted_words = calculate_weighted_words_for_topic_scopes(
    topic_scopes=topic_scopes,
    documents=documents,
    document_topics=document_topics, # type: ignore
    interpreter=interpreter
  )
  for topic_id, topic_words in merged_topic_weighted_words.items():
    merged_topic = Topic(
      id=topic_id,
      children=[],
      frequency=0,
      label=None,
      words=topic_words
    )
    merged_topics.append(merged_topic)

  topics_locker: dict[int, Topic] = {}
  for topic in base_topics:
    topics_locker[topic.id] = topic
  for topic in merged_topics:
    topics_locker[topic.id] = topic
  
  topic_hierarchy = topics_locker[hierarchy_root]
  topic_hierarchy_layers = list(nx.bfs_layers(hierarchy, hierarchy_root))[:-1]
  for layer in topic_hierarchy_layers:
    for parent in layer:
      for child in hierarchy.successors(parent):
        if topics_locker[parent].children is None:
          topics_locker[parent].children = []
        topics_locker[parent].children.append(topics_locker[child]) # type: ignore

  topic_hierarchy.recalculate_frequency()

  return topic_hierarchy