import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import os

# Load evaluation scores
scores = pd.read_csv("results/clustering_evaluation.csv", index_col=0)

# Load feature representations
def load_features(method="tfidf", dataset="news"):
    """Load feature representations for Newsgroups or Wikipedia People datasets."""
    # Adjust path to match your actual data structure
    if dataset == "news":
        file_path = f"data/vectorization_embedding_data/{method}_newsgroups.csv"
    else:  # wiki
        file_path = f"data/vectorization_embedding_data/{method}_people_wiki.csv"
    
    print(f"Loading features from: {file_path}")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature file not found: {file_path}")
    
    # Load based on the method type
    if method == "tfidf":
        return pd.read_csv(file_path).values
    else:  # word2vec or glove
        return pd.read_csv(file_path, header=None).values

# Load clustering labels
def load_cluster_labels(method="kmeans", representation="tfidf_news"):
    """Load cluster labels produced by clustering algorithms."""
    # Adjust path to match your actual data structure
    file_path = f"results/Cluster/{method}_{representation}.csv"
    
    print(f"Loading cluster labels from: {file_path}")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cluster labels file not found: {file_path}")
        
    return pd.read_csv(file_path).values.flatten()

# 🟢 **1️⃣ t-SNE & PCA Visualization**
def visualize_tsne_pca(method="kmeans", representation="tfidf_news"):
    """Visualize clustering results using t-SNE and PCA."""
    print(f"🔹 Visualizing {method.upper()} on {representation.upper()} using t-SNE & PCA...")

    try:
        # Split representation into method and dataset
        rep_method, dataset = representation.split("_")
        
        # Load data and cluster labels
        data = load_features(rep_method, dataset)
        labels = load_cluster_labels(method, representation)
        
        # Check if the dimensions match
        if len(data) != len(labels):
            print(f"⚠️ Dimension mismatch: Features: {len(data)}, Clusters: {len(labels)}")
            print(f"🔧 Using the first {min(len(data), len(labels))} samples for visualization.")
            
            # Use the smaller dimension
            min_len = min(len(data), len(labels))
            data = data[:min_len]
            labels = labels[:min_len]

        # Apply dimensionality reduction on a sample if data is too large
        if len(data) > 10000:
            sample_size = 10000
            print(f"⚠️ Large dataset detected. Using a random sample of {sample_size} points for visualization.")
            indices = np.random.choice(len(data), sample_size, replace=False)
            data_sample = data[indices]
            labels_sample = labels[indices]
        else:
            data_sample = data
            labels_sample = labels

        # Apply PCA
        print("Computing PCA...")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_sample)

        # Apply t-SNE
        print("Computing t-SNE (this may take a while)...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_result = tsne.fit_transform(data_sample)

        # Create results directory if it doesn't exist
        os.makedirs("results/visualizations", exist_ok=True)

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # PCA Visualization
        scatter1 = sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], 
                               hue=labels_sample, palette="coolwarm", ax=axes[0])
        axes[0].set_title(f"{method.upper()} Clusters (PCA)")
        
        # Limit legend items if there are too many clusters
        if len(np.unique(labels_sample)) > 10:
            scatter1.legend_.remove()
            axes[0].set_title(f"{method.upper()} Clusters (PCA) - {len(np.unique(labels_sample))} clusters")

        # t-SNE Visualization
        scatter2 = sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], 
                                hue=labels_sample, palette="coolwarm", ax=axes[1])
        axes[1].set_title(f"{method.upper()} Clusters (t-SNE)")
        
        # Limit legend items if there are too many clusters
        if len(np.unique(labels_sample)) > 10:
            scatter2.legend_.remove()
            axes[1].set_title(f"{method.upper()} Clusters (t-SNE) - {len(np.unique(labels_sample))} clusters")

        plt.tight_layout()
        output_path = f"results/visualizations/{method}_{representation}_tsne_pca.png"
        plt.savefig(output_path, dpi=300)
        plt.close()  # Close the figure to free memory

        print(f"✅ Saved t-SNE & PCA visualization to {output_path}")
        
    except Exception as e:
        print(f"❌ Error visualizing {method} on {representation}: {str(e)}")

# 🟢 **2️⃣ Dendrogram Visualization (for smaller datasets only)**
def visualize_dendrogram(representation="tfidf_news"):
    """Generate and plot a dendrogram for hierarchical clustering."""
    print(f"🔹 Creating Dendrogram for Hierarchical Clustering on {representation.upper()}...")

    try:
        # Split representation into method and dataset
        rep_method, dataset = representation.split("_")
        
        # Load feature data
        data = load_features(rep_method, dataset)
        
        # Check if data is too large for dendrogram
        if len(data) > 1000:
            print(f"⚠️ Dataset too large ({len(data)} samples) for complete dendrogram.")
            print(f"🔧 Using a random sample of 1000 points.")
            indices = np.random.choice(len(data), 1000, replace=False)
            data = data[indices]
        
        # Create results directory if it doesn't exist
        os.makedirs("results/visualizations", exist_ok=True)

        # Compute hierarchical linkage matrix
        print("Computing linkage matrix (this may take a while)...")
        linked = linkage(data, method="ward")

        # Plot dendrogram
        plt.figure(figsize=(12, 6))
        dendrogram(linked, orientation="top", distance_sort="descending", 
                  show_leaf_counts=True, truncate_mode='lastp', p=30)
        plt.title(f"Dendrogram for {representation.upper()}")
        plt.xlabel("Clusters")
        plt.ylabel("Distance")

        output_path = f"results/visualizations/dendrogram_{representation}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()  # Close the figure to free memory

        print(f"✅ Dendrogram saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Error creating dendrogram for {representation}: {str(e)}")

# **Run visualizations**
if __name__ == "__main__":
    print("🎨 Starting visualization process...")
    
    # Create a list of key representation-method pairs that worked in evaluation
    visualization_targets = [
        # TF-IDF representations
        ("kmeans", "tfidf_news"),
        ("hierarchical", "tfidf_news"),
        ("gmm", "tfidf_news"),
        ("kmeans", "tfidf_wiki"),
        ("hierarchical", "tfidf_wiki"),
        ("gmm", "tfidf_wiki"),
        
        # Word2Vec wiki representations (News had issues)
        ("kmeans", "word2vec_wiki"),
        ("hierarchical", "word2vec_wiki"),
        ("gmm", "word2vec_wiki"),
        
        # GloVe wiki representations (News had issues)
        ("kmeans", "glove_wiki"),
        ("hierarchical", "glove_wiki"),
        ("gmm", "glove_wiki")
    ]
    
    # Run t-SNE & PCA for each target
    for method, representation in visualization_targets:
        visualize_tsne_pca(method, representation)
    
    # Run dendrograms for smaller datasets
    dendro_targets = ["tfidf_news", "tfidf_wiki", "word2vec_wiki", "glove_wiki"]
    for representation in dendro_targets:
        visualize_dendrogram(representation)

    print("🏁 All visualizations completed!")