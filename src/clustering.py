from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd

# 🔹 Loading features saved during feature_extraction
def load_features():
    """
    Load TF-IDF, Word2Vec, and GloVe features from saved files
    """
    
    print("🔄 Loading extracted features...")
    
    # ✅ Loading TF-IDF features
    tfidf_news = pd.read_csv('data/vectorization_embedding_data/tfidf_newsgroups.csv').values
    tfidf_wiki = pd.read_csv('data/vectorization_embedding_data/tfidf_people_wiki.csv').values
    
    # ✅ Loading Word2Vec features
    word2vec_news = pd.read_csv('data/vectorization_embedding_data/word2vec_newsgroups.csv', header=None).values
    word2vec_wiki = pd.read_csv('data/vectorization_embedding_data/word2vec_people_wiki.csv', header=None).values
    
    # ✅ Loading features GloVe
    glove_news = pd.read_csv('data/vectorization_embedding_data/glove_newsgroups.csv', header=None).values
    glove_wiki = pd.read_csv('data/vectorization_embedding_data/glove_people_wiki.csv', header=None).values
    
    print("✅ Features successfully loaded!")
    
    return tfidf_news, tfidf_wiki, word2vec_news, word2vec_wiki, glove_news, glove_wiki

# K-Means application
def apply_kmeans(data, num_clusters=3):
    kmeans = KMeans(init='k-means++',n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return kmeans, labels

# Hierarchical Clustering application
def apply_hierarchical(data, num_clusters=10):
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
    labels = hierarchical.fit_predict(data)
    return hierarchical, labels

# Gaussian Mixture Model (GMM) application
def apply_gmm(data, num_clusters=10):
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    labels = gmm.fit_predict(data)
    return gmm, labels

# Latent Dirichlet Allocation (LDA) application
def apply_lda(data, num_topics=10):
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    topic_distributions = lda.fit_transform(data)
    return lda, topic_distributions

def save_results(labels, filename):
    """
    Save results to a CSV file
    """
    pd.DataFrame(labels).to_csv(f'results/Cluster/{filename}.csv', index=False)

if __name__ == "__main__":
    # Load feature
    tfidf_news, tfidf_wiki, word2vec_news, word2vec_wiki, glove_news, glove_wiki = load_features()
    
    # Define which clustering methods to apply to each representation
    # General methods for all representation types
    general_methods = {
        "kmeans": apply_kmeans,
        "hierarchical": apply_hierarchical,
        "gmm": apply_gmm
    }
    
    # LDA method only for TF-IDF data
    tfidf_methods = {
        "kmeans": apply_kmeans,
        "hierarchical": apply_hierarchical,
        "gmm": apply_gmm,
        "lda": apply_lda
    }
    
    # Representations
    tfidf_representations = {
        "tfidf_news": tfidf_news,
        "tfidf_wiki": tfidf_wiki
    }
    
    embedding_representations = {
        "word2vec_news": word2vec_news,
        "word2vec_wiki": word2vec_wiki,
        "glove_news": glove_news,
        "glove_wiki": glove_wiki
    }
    
    # Storing clustering results
    clustering_results = {}
    
    # Apply all methods (including LDA) to TF-IDF representations
    for rep_name, rep_data in tfidf_representations.items():
        print(f"🔹 Applying clustering algorithms to {rep_name}...")
        
        for method_name, method_func in tfidf_methods.items():
            try:
                model, labels = method_func(rep_data)  # Apply algorithm
                clustering_results[f"{method_name}_{rep_name}"] = labels  # save results
                print(f"✅ {method_name.upper()} applied to {rep_name}.")
            except Exception as e:
                print(f"⚠️ Error applying {method_name.upper()} to {rep_name}: {str(e)}")
    
    # Apply only general methods (excluding LDA) to embedding representations
    for rep_name, rep_data in embedding_representations.items():
        print(f"🔹 Applying clustering algorithms to {rep_name}...")
        
        for method_name, method_func in general_methods.items():
            try:
                model, labels = method_func(rep_data)  # Apply algorithm
                clustering_results[f"{method_name}_{rep_name}"] = labels  # save results
                print(f"✅ {method_name.upper()} applied to {rep_name}.")
            except Exception as e:
                print(f"⚠️ Error applying {method_name.upper()} to {rep_name}: {str(e)}")
    
    # Save results
    for name, labels in clustering_results.items():
        try:
            save_results(labels, name)
        except Exception as e:
            print(f"⚠️ Error saving results for {name}: {str(e)}")
    
        

    
    
    
    print("✅ Clustering completed and results saved!")

