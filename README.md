# ML2_Project
# Document Clustering Project

## Overview

This project aims to apply unsupervised learning techniques to cluster documents from two datasets: the **People Wikipedia Dataset** and the **20 Newsgroups Dataset**. The goal is to uncover inherent structures within these datasets, providing insights into natural groupings of the documents.

## Datasets

### 1. People Wikipedia Dataset

**Description**:  
The **People Wiki Dataset** consists of biographical articles of notable individuals extracted from Wikipedia. Each entry contains **a unique URI, the person's name, and text extracted from their Wikipedia page**. The dataset allows us to analyze relationships between individuals based on the content of their biographies, such as similarities in professions, historical relevance, and other contextual attributes.

**Source**:  
This dataset is derived from Wikipedia through structured data extraction techniques.

**Features**:
- **URI**: A unique identifier (Uniform Resource Identifier) for each person’s Wikipedia page.
- **Name**: Full name of the individual.
- **Text**: Extracted content from their Wikipedia biography, which provides contextual information about their profession, achievements, and background.

### 2. 20 Newsgroups Dataset

**Description**:  
This dataset contains approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups. The content covers a wide range of topics, from politics and religion to sports and technology.

**Source**:  
The dataset is widely used for experiments in text applications of machine learning techniques, such as text classification and clustering.

**Features**:
- **Content**: The main body of the newsgroup post.
- **Headers**: Metadata such as subject line, author, and date.

## Methodology

The project will follow these key steps:

1. **Data Collection**:  
   - Obtain the datasets from their respective sources.

2. **Data Preprocessing**:
   - **Cleaning**: Remove noise, handle missing values, and normalize text (e.g., lowercasing, stemming).
   - **Tokenization**: Split text into tokens (words or phrases).
   - **Stop Words Removal**: Eliminate common words that may not contribute to clustering (e.g., "the", "and").

3. **Feature Extraction**:
   - **TF-IDF Vectorization**: Convert textual data into numerical features using Term Frequency-Inverse Document Frequency.
   - **Word Embeddings**: Apply techniques like Word2Vec or GloVe to capture semantic relationships. [ Optional ]

4. **Clustering Algorithms**:
   - **K-Means Clustering**: Partition documents into 'k' clusters based on feature similarity.
   - **Hierarchical Clustering**: Create a tree-like structure to represent nested groupings of documents.
   - **Latent Dirichlet Allocation (LDA)**: Identify underlying topics within the documents and group them accordingly. [ Optional ]

5. **Evaluation of Clusters**:
   - **Silhouette Score**: Measure how similar a document is to its own cluster compared to other clusters.
   - **Purity Score**: Assess the extent to which clusters contain a single category of documents.

6. **Visualization**:
   - **t-SNE or PCA**: Reduce dimensionality for visual representation of clusters.
   - **Dendrograms**: Visualize the arrangement of clusters in hierarchical clustering.

## Tools and Technologies

- **Programming Languages**: Python
- **Libraries**:
  - **Data Manipulation**: pandas, NumPy
  - **Text Processing**: NLTK, spaCy
  - **Machine Learning**: scikit-learn, gensim
  - **Visualization**: matplotlib, seaborn

## Deliverables

Students are expected to submit the following:

1. **Structured Code**:
   - The project code should be structured into clear, reusable modules.
   - Recommended folder structure:
     ```
     ├── data/                  # Folder for datasets (raw and preprocessed)
     ├── src/
     │   ├── preprocessing.py    # Code for data cleaning and preprocessing
     │   ├── feature_extraction.py  # Code for vectorization and embedding
     │   ├── clustering.py       # Code for implementing clustering models
     │   ├── evaluation.py       # Code for computing clustering metrics
     │   ├── visualization.py    # Code for plotting results
     │   ├── main.py             # Main script to run the project pipeline
     ├── notebooks/              # Jupyter notebooks for exploratory data analysis
     ├── results/                # Folder to save cluster results and visualizations
     ├── requirements.txt        # List of required Python libraries
     ├── README.md               # Project documentation
     ```
   - Each script should be well-commented and modularized.
   - Code should follow best practices in Python programming.

2. **Evaluation Metrics**: Quantitative assessments of the clustering performance, including relevant metrics.
3. **Visualizations**: Graphs and plots illustrating the clustering results and any notable patterns or insights.


## References

- **People Wikipedia**: A project extracting structured content from Wikipedia. [Link](https://www.kaggle.com/datasets/sameersmahajan/people-wikipedia-data)
- **20 Newsgroups Dataset**: A collection of approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups. [Link](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups)
