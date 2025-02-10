def bertopic_visualization_embeddings(reduced_embeddings: npt.NDArray, topic_assignments: npt.NDArray, topics: int)->BERTopicVisualizationEmbeddingResult:
  

  from umap import UMAP
  high_dimensional_embeddings = np.vstack([reduced_embeddings, topic_embeddings])
  umap_model = UMAP(
    n_neighbors=column.topic_modeling.globality_consideration or column.topic_modeling.min_topic_size,
    min_dist=0.1,
    n_components=2
  )
  visualization_embeddings = umap_model.fit_transform(high_dimensional_embeddings)
  low_document_embeddings = visualization_embeddings[:len(reduced_embeddings)]
  low_topic_embeddings = visualization_embeddings[len(reduced_embeddings):]

  return BERTopicVisualizationEmbeddingResult(
    document_embeddings=document_embeddings,
    topic_embeddings=topic_embeddings
  )
