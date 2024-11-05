# What is this?

This application aims to help people discover common themes in a large dataset by leveraging the power of BERTopic, which is explained in [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) and implemented in [https://maartengr.github.io/BERTopic/index.html](https://maartengr.github.io/BERTopic/index.html). Rather than dealing with documents (pieces of text) as they were, this application receives tabular datasets like CSV or EXCEL that contains textual columns so that the topics discovered by BERTopic can be used for further analysis.

The application allows users to discover relationships between the topics and other variables/columns in the dataset using interactive plots provided by Plotly, so that users can potentially gain more insights about the dataset.

# How to Run Application

**Prerequisites**:
- Make sure that you have Python 3.10+ installed in your system, preferably the standard distribution from [https://python.org](https://python.org), as we cannot guarantee that the application will work in other environments.


**Setup**:
- Get a copy of this repository. Run ``git clone https://github.com/DavinTristanIeson/Skripsi`` in the directory where you want to have the project if you have Git; or just download a copy of it from GitHub and unzip it in your folder of choice.
- Run ``python scripts/setup.py`` from the same directory as this file (README.md). If you're a developer and wants to mess around with the dependencies in requirements.in, use ``python scripts/setup.py --dev``
- Run ``python scripts/download_interface.py`` to download the default front-end interface.

**Run App**:

If all goes well, you should be able to run the application by running ``python app.py``. The interface will be accessible through ``http://localhost:8000``.

# Future Directions
Implemented features:
- Users can extract topics from the dataset using BERTopic, paired with SBERT, Doc2Vec, or TF-IDF embeddings (topic modeling provided by ``bertopic``, preprocessing provided by ``spacy`` and ``gensim``).
- Users can customize preprocessing and topic modeling parameters.
- Users can view topic words, document clusters, and intertopic relationship (visualizations provided by ``bertopic`` with the help of ``plotly``)
- Users can evaluate the quality of the topics (topic coherence calculation provided by ``gensim``)
- Users can view the relationship between the topics and other variables (be they textual, categorical, continuous, or temporal) in the dataset (visualization provided by ``plotly``)

Planned, but not implemented features:
- Users can view and filter the dataset.
- Users can view a statistical summary of the current filtered dataset.
- Users can see the documents that are categorized as part of a topic, with the topic words highlighted.
- Users can view the statistical significance of the association between the topics and other variables.