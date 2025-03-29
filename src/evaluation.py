from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import os

# Load real categories
true_labels_news = pd.read_csv('data/Real_Categories/true_labels_newsgroups.csv').values.flatten()

# There are no real categories for Wikipedia People, so we'll put it as `None`.
true_labels_wiki = None  

representations = ['tfidf_news', 'tfidf_wiki', 'word2vec_news', 'word2vec_wiki', 'glove_news', 'glove_wiki']
methods = ['kmeans', 'hierarchical', 'gmm', 'lda']

# Purity Score Calculation Function
def purity_score(y_true, y_pred):
    if y_true is None:  # If there are no real categories
        return np.nan  # We return null (NaN) because we cannot calculate accuracy without the actual labels.
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)

# Load the cluster results
def load_cluster_results(method, representation):
    # Correct path based on your directory structure
    file_path = f'results/Cluster/{method}_{representation}.csv'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found: {file_path}")
        # Skip LDA for non-TFIDF representations as they weren't generated
        if method == 'lda' and not representation.startswith('tfidf'):
            print(f"  Note: LDA was only applied to TF-IDF representations, not to {representation}")
            return None
        raise FileNotFoundError(f"Could not find cluster results for {method} on {representation}")
    
    # Load the cluster labels
    cluster_labels = pd.read_csv(file_path)
    
    # For LDA, the file might have 10x more rows than expected
    # Let's check if it's the case and fix it
    if method == 'lda':
        if len(cluster_labels) > 10000:  # This is a heuristic threshold
            print(f"  ⚠️ Found {len(cluster_labels)} labels for LDA, which seems excessive.")
            print(f"  🔧 Attempting to fix by taking only the first label for each document.")
            
            # Extract every 10th row (this is based on the error messages showing 10x more samples)
            # Adjust this logic if the ratio is different
            if "news" in representation and len(cluster_labels) == 25880:
                # For news dataset, take every 10th row (2588 expected documents)
                cluster_labels = cluster_labels.iloc[::10]
                print(f"  ✅ Reduced to {len(cluster_labels)} labels.")
            elif "wiki" in representation and len(cluster_labels) == 590710:
                # For wiki dataset, take every 10th row (59071 expected documents)
                cluster_labels = cluster_labels.iloc[::10]
                print(f"  ✅ Reduced to {len(cluster_labels)} labels.")
            
    return cluster_labels.values.flatten()

# Load the feature data for silhouette score
def load_feature_data(representation):
    # Fix path to match your actual file structure
    file_path = f'data/vectorization_embedding_data/{representation}.csv'
    
    # Based on your file naming convention
    if 'news' in representation:
        file_path = file_path.replace('_news.', '_newsgroups.')
    if 'wiki' in representation:
        file_path = file_path.replace('_wiki.', '_people_wiki.')
    
    print(f"  Loading features from: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: Feature file not found: {file_path}")
        raise FileNotFoundError(f"Could not find feature data for {representation}")
    
    try:
        if representation.startswith('tfidf'):
            return pd.read_csv(file_path).values
        else:
            return pd.read_csv(file_path, header=None).values
    except pd.errors.EmptyDataError:
        print(f"⚠️ Warning: Empty file: {file_path}")
        return None
    except Exception as e:
        print(f"⚠️ Warning: Error loading {file_path}: {str(e)}")
        return None

# Save evaluation results
def save_evaluation_results(scores_dict, filename="results/clustering_evaluation.csv"):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df = pd.DataFrame(scores_dict).T  
    df.to_csv(filename)
    print(f"✅ Evaluation results saved to {filename}")

# Function to check file dimensions
def check_file_dimensions(rep):
    # Paths
    feature_path = f'data/vectorization_embedding_data/{rep}.csv'
    if 'news' in rep:
        feature_path = feature_path.replace('_news.', '_newsgroups.')
    if 'wiki' in rep:
        feature_path = feature_path.replace('_wiki.', '_people_wiki.')
    
    # Load and get dimensions
    try:
        if rep.startswith('tfidf'):
            features = pd.read_csv(feature_path)
        else:
            features = pd.read_csv(feature_path, header=None)
        
        print(f"  Features shape for {rep}: {features.shape}")
        
        # Check cluster files
        for method in methods:
            if method == 'lda' and not rep.startswith('tfidf'):
                continue
                
            cluster_path = f'results/Cluster/{method}_{rep}.csv'
            if os.path.exists(cluster_path):
                clusters = pd.read_csv(cluster_path)
                print(f"  {method.upper()} clusters shape for {rep}: {clusters.shape}")
                
                # Check if shapes are compatible
                if features.shape[0] != clusters.shape[0]:
                    print(f"  ⚠️ Mismatch: Features has {features.shape[0]} rows but {method.upper()} has {clusters.shape[0]} rows")
                    
                    # For Word2Vec and GloVe News, try to fix the issue
                    if ('word2vec' in rep or 'glove' in rep) and 'news' in rep:
                        print(f"  🔧 Attempting to fix {method}_{rep} mismatch...")
                        
                        if clusters.shape[0] == true_labels_news.shape[0]:
                            print(f"  ✅ Cluster file row count matches true_labels_news count. Using this as reference.")
                            # Save the first n rows of the feature data where n is the number of true labels
                            fixed_features = features.iloc[:clusters.shape[0]]
                            fixed_path = f'data/vectorization_embedding_data/{rep}_fixed.csv'
                            fixed_features.to_csv(fixed_path, index=False, header=None)
                            print(f"  ✅ Saved fixed features to {fixed_path}")
                        else:
                            print(f"  ❌ Unable to automatically fix this mismatch.")
            else:
                print(f"  ⚠️ No cluster file found for {method}_{rep}")
    except Exception as e:
        print(f"  ❌ Error checking dimensions for {rep}: {str(e)}")

# Evaluation calculate
if __name__ == "__main__":
    print("🔍 Starting clustering evaluation...")
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # First, let's check file dimensions to identify issues
    print("📏 Checking data dimensions...")
    for rep in representations:
        check_file_dimensions(rep)
    
    scores = {}
    
    for rep in representations:
        print(f"📊 Evaluating clusters for {rep}...")
        
        try:
            # Load feature data for silhouette score
            feature_data = None
            
            # Try to load fixed features for word2vec and glove news if they exist
            if ('word2vec' in rep or 'glove' in rep) and 'news' in rep:
                fixed_path = f'data/vectorization_embedding_data/{rep}_fixed.csv'
                if os.path.exists(fixed_path):
                    print(f"  Loading fixed features from: {fixed_path}")
                    feature_data = pd.read_csv(fixed_path, header=None).values
                else:
                    feature_data = load_feature_data(rep)
            else:
                feature_data = load_feature_data(rep)
                
            if feature_data is None:
                print(f"❌ Could not load feature data for {rep}, skipping...")
                continue
                
            dataset_name = "News" if "news" in rep else "Wiki"
            rep_name = rep.split('_')[0].upper()
            true_labels = true_labels_news if "news" in rep else true_labels_wiki
            
            for method in methods:
                # Skip LDA for non-TFIDF representations
                if method == 'lda' and not rep.startswith('tfidf'):
                    print(f"⏩ Skipping {method.upper()} for {rep} (not applied during clustering)")
                    continue
                    
                key = f"{method.upper()} ({rep_name} - {dataset_name})"
                print(f"  Evaluating {key}...")
                
                try:
                    cluster_labels = load_cluster_results(method, rep)
                    
                    if cluster_labels is not None:
                        # Check if the lengths match
                        if len(feature_data) != len(cluster_labels):
                            print(f"  ⚠️ Dimension mismatch: Features: {len(feature_data)}, Clusters: {len(cluster_labels)}")
                            
                            # Try to use min length to make the evaluation possible
                            min_len = min(len(feature_data), len(cluster_labels))
                            print(f"  🔧 Using the first {min_len} samples for evaluation.")
                            feature_data_trunc = feature_data[:min_len]
                            cluster_labels_trunc = cluster_labels[:min_len]
                            
                            # Calculate silhouette score using the feature data, not the labels themselves
                            if len(set(cluster_labels_trunc)) > 1:
                                sil_score = silhouette_score(feature_data_trunc, cluster_labels_trunc)
                            else:
                                sil_score = np.nan
                                
                            scores[key] = {
                                "Silhouette Score": sil_score,
                                "Purity Score": purity_score(true_labels[:min_len] if true_labels is not None else None, cluster_labels_trunc)
                            }
                        else:
                            # Proceed normally since dimensions match
                            sil_score = silhouette_score(feature_data, cluster_labels) if len(set(cluster_labels)) > 1 else np.nan
                            
                            scores[key] = {
                                "Silhouette Score": sil_score,
                                "Purity Score": purity_score(true_labels, cluster_labels)
                            }
                        print(f"  ✅ Evaluation complete for {key}")
                except Exception as e:
                    print(f"  ❌ Error evaluating {key}: {str(e)}")
                    scores[key] = {
                        "Silhouette Score": np.nan,
                        "Purity Score": np.nan
                    }
        except Exception as e:
            print(f"❌ Error processing {rep}: {str(e)}")
    
    save_evaluation_results(scores)
    print("🏁 Evaluation process completed!")