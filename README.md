# What is this?

**TopicTrail** aims to help people discover topics in a large dataset by leveraging the power of BERTopic, which is explained in [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) and implemented in [https://maartengr.github.io/BERTopic/index.html](https://maartengr.github.io/BERTopic/index.html). Rather than dealing with documents (pieces of text) as they were, this application receives tabular datasets like CSV or EXCEL that contains textual columns so that the topics discovered by BERTopic can be used for further analysis.

For example, one might be inclined to check if a particular topic is associated with a higher rating (e.g.: customer review datasets), or is more associated with a particular field (e.g.: abstracts dataset). This association may potentially reveal more insights about the topics and its relationships.

# How to Run the Application

**Prerequisites**:

- Make sure that you have Python 3.10+ installed in your system, preferably the standard distribution from [https://python.org](https://python.org), as we cannot guarantee that the application will work in other environments.

**Setup**:

- Get a copy of this repository. Run `git clone https://github.com/DavinTristanIeson/Skripsi` in the directory where you want to have the project if you have Git; or just download a copy of it from GitHub and unzip it in your folder of choice.
- Run `python scripts/setup.py` from the same directory as this file (README.md). If you're a developer and wants to mess around with the dependencies in requirements.in, use `python scripts/setup.py --dev` to sync `requirements.lock` with `requirements.in`.
- Run `python scripts/download_interface.py` to download the default front-end interface which is hosted in `https://github.com/DavinTristanIeson/Skripsi-Frontend`.

**Run App**:

If all goes well, you should be able to run the application by running the following commands:

- If you are in Windows, run start.bat with `./start.bat`.
- If you are in MacOS or Linux, run start.sh with `./start.sh` or `bash ./start.sh`.
- Alternatively you can also manually start the application by activating the virtual environment `venv/scripts/activate` and then calling
  `fastapi run`. The interface will be accessible through `http://localhost:8000`.

# Features

**What TopicTrail is**: TopicTrail can be used to quickly extract insights from a dataset without having to set up a dedicated environment to perform topic modeling and regression analysis. It is intended to be used for quick and simple exploration of a dataset (that contains textual data).

**What TopicTrail isn't**: TopicTrail is not meant to perform more complex analysis such as performing regression with other independent variables besides binary independent variables; or using more complex regression models like GLM or ARIMA. Even so, the results of TopicTrail is always stored inside the `data/` folder inside the project directory; so analysts can simply load `workspace.parquet` containing the modified dataset, or the topic information inside `topic-modeling/<column>/topics.json`, or the BERTopic model itself inside `topic-modeling/<column>/bertopic`.

Existing features:

- Users can extract topics from the dataset using BERTopic and view the extracted topics through an interface.
- Users can refine the topics extracted via the c-TF-IDF procedure of BERTopic by modifying the document-topic assignments.
- Users can add metadata to each topic to aid interpretation.
- Users can customize preprocessing and topic modeling parameters.
- Users can evaluate the quality of the topics and optimize the hyperparameters of HDBSCAN (part of BERTopic) to maximize coherence.
- Users can quickly view and compare subdatasets (subsets of a dataset) in the interface through graphs and plots to get a quick overview of the data.
- Users can model the relationship between the topics (or any binary independent variables) with other dependent variables in the dataset with regression models.

# Contributors

This project is primary developed and maintained by:

- **Davin Tristan Ieson** - Lead Developer

Additional contributions by:

- **Angeline Ho** - Quality Control
- **Hansen Tanio** - UI/UX Designer
