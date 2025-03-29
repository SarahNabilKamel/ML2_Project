from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import pandas as pd
import pickle
import numpy as np
import os


# Load cleaned data from preprocessing.py
def load_cleaned_data():
    print("📥 Loading Cleaned Datasets...")
    newsgroups_df = pd.read_csv('data/Preprocessed_Cleand_Data/cleaned_newsgroups.csv')
    people_wiki_df = pd.read_csv('data/Preprocessed_Cleand_Data/cleaned_people_wiki.csv')
    
    # Check for NaN values
    news_nan_count = newsgroups_df['cleaned_text'].isna().sum()
    wiki_nan_count = people_wiki_df['cleaned_text'].isna().sum()
    
    if news_nan_count > 0:
        print(f"⚠️ Found {news_nan_count} NaN values in Newsgroups dataset. Filling with empty strings.")
        newsgroups_df['cleaned_text'] = newsgroups_df['cleaned_text'].fillna('')
    
    if wiki_nan_count > 0:
        print(f"⚠️ Found {wiki_nan_count} NaN values in People Wiki dataset. Filling with empty strings.")
        people_wiki_df['cleaned_text'] = people_wiki_df['cleaned_text'].fillna('')
    
    # Check for empty strings and give a warning
    news_empty_count = (newsgroups_df['cleaned_text'] == '').sum()
    wiki_empty_count = (people_wiki_df['cleaned_text'] == '').sum()
    
    if news_empty_count > 0:
        print(f"⚠️ Found {news_empty_count} empty documents in Newsgroups dataset.")
    
    if wiki_empty_count > 0:
        print(f"⚠️ Found {wiki_empty_count} empty documents in People Wiki dataset.")
    
    print("✅ Loading Datasets is done...")
    return newsgroups_df, people_wiki_df


# Feature extraction using TF-IDF
def extract_tfidf_features(texts, max_features=3):
    print(f"🔍 Extracting TF-IDF features (max_features={max_features})...")
    
    # Make sure all texts are strings
    texts = texts.astype(str)
    
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    print(f"✅ TF-IDF extraction complete. Matrix shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix


# Feature extraction using Word2Vec
def extract_word2vec_features(texts, vector_size=500, window=5, min_count=2):
    print(f"🔍 Extracting Word2Vec features (vector_size={vector_size})...")
    
    # Ensure all texts are strings
    texts = texts.astype(str)
    
    tokenized_texts = [text.split() for text in texts]
    model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    word_vectors = model.wv
    
    # Represent each text as its word vectors
    document_vectors = []
    empty_vectors = 0
    
    for tokens in tokenized_texts:
        vectors = [word_vectors[word] for word in tokens if word in word_vectors]
        if vectors:
            document_vectors.append(np.mean(vectors, axis=0))
        else:
            document_vectors.append(np.zeros(vector_size))
            empty_vectors += 1
    
    if empty_vectors > 0:
        print(f"⚠️ {empty_vectors} documents have no vectors in Word2Vec model (using zero vectors instead)")
    
    print(f"✅ Word2Vec extraction complete. Matrix shape: {np.array(document_vectors).shape}")
    return model, np.array(document_vectors)


# Load the GloVe model and convert it to a dictionary
def load_glove_model(glove_path="models/glove.6B.300d.txt"):
    print("📥 Loading GloVe model...")
    
    # Check if file exists
    if not os.path.exists(glove_path):
        print(f"❌ GloVe model file not found at {glove_path}")
        print("Skipping GloVe feature extraction")
        return None
    
    glove_dict = {}
    try:
        with open(glove_path, encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                glove_dict[word] = vector
        print(f"✅ GloVe model loaded successfully! Vocabulary size: {len(glove_dict)}")
        return glove_dict
    except Exception as e:
        print(f"❌ Error loading GloVe model: {e}")
        return None


# Feature extraction using GloVe
def extract_glove_features(texts, glove_dict, vector_size=300):
    if glove_dict is None:
        print("❌ GloVe dictionary is None, skipping feature extraction")
        return np.zeros((len(texts), vector_size))
    
    print(f"🔍 Extracting GloVe features (vector_size={vector_size})...")
    
    # Ensure all texts are strings
    texts = texts.astype(str)
    
    document_vectors = []
    empty_vectors = 0
    
    for text in texts:
        words = text.split()
        word_vectors = [glove_dict[word] for word in words if word in glove_dict]
        
        if word_vectors:
            document_vector = np.mean(word_vectors, axis=0)
        else:
            document_vector = np.zeros(vector_size)
            empty_vectors += 1
        
        document_vectors.append(document_vector)
    
    if empty_vectors > 0:
        print(f"⚠️ {empty_vectors} documents have no vectors in GloVe model (using zero vectors instead)")
    
    print(f"✅ GloVe extraction complete. Matrix shape: {np.array(document_vectors).shape}")
    return np.array(document_vectors)


# 🔹 Saving models and extracted data
def save_model(obj, filename):
    """
    Save any object (model, matrix, etc.) using pickle
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    print(f"💾 Saving model to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✅ Model saved successfully")


def save_matrix(matrix, filename):
    """
    Save matrices to CSV files
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    print(f"💾 Saving matrix to {filename}")
    df = pd.DataFrame(matrix)
    df.to_csv(filename, index=False)
    print(f"✅ Matrix saved successfully")


if __name__ == "__main__":
    print("🚀 Starting feature extraction process...")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/vectorization_embedding_data", exist_ok=True)
    
    # Load data
    newsgroups_df, people_wiki_df = load_cleaned_data()
    
    # TF-IDF application to texts
    vectorizer_news, tfidf_news = extract_tfidf_features(newsgroups_df['cleaned_text'])
    vectorizer_wiki, tfidf_wiki = extract_tfidf_features(people_wiki_df['cleaned_text'])
    
    # Word2Vec application to texts
    word2vec_model_news, word2vec_news = extract_word2vec_features(newsgroups_df['cleaned_text'])
    word2vec_model_wiki, word2vec_wiki = extract_word2vec_features(people_wiki_df['cleaned_text'])

    # Load the GloVe model
    glove_dict = load_glove_model()

    # Extract GloVe features if model loaded successfully
    if glove_dict is not None:
        glove_news = extract_glove_features(newsgroups_df['cleaned_text'], glove_dict)
        glove_wiki = extract_glove_features(people_wiki_df['cleaned_text'], glove_dict)
        
        # Save GloVe matrices
        save_matrix(glove_news, 'data/vectorization_embedding_data/glove_newsgroups.csv')
        save_matrix(glove_wiki, 'data/vectorization_embedding_data/glove_people_wiki.csv')
    
    # Save TF-IDF models
    save_model(vectorizer_news, 'models/tfidf_vectorizer_news.pkl')
    save_model(vectorizer_wiki, 'models/tfidf_vectorizer_wiki.pkl')
    
    # Save TF-IDF matrices
    save_matrix(tfidf_news.toarray(), 'data/vectorization_embedding_data/tfidf_newsgroups.csv')
    save_matrix(tfidf_wiki.toarray(), 'data/vectorization_embedding_data/tfidf_people_wiki.csv')
    
    # Save Word2Vec forms and matrices
    word2vec_model_news.save('models/word2vec_news.model')
    word2vec_model_wiki.save('models/word2vec_wiki.model')
    save_matrix(word2vec_news, 'data/vectorization_embedding_data/word2vec_newsgroups.csv')
    save_matrix(word2vec_wiki, 'data/vectorization_embedding_data/word2vec_people_wiki.csv')

    print("\n✅ Feature extraction process completed successfully!")
    print("📊 Summary of extracted features:")
    print(f"- TF-IDF Newsgroups: {tfidf_news.shape}")
    print(f"- TF-IDF People Wiki: {tfidf_wiki.shape}")
    print(f"- Word2Vec Newsgroups: {word2vec_news.shape}")
    print(f"- Word2Vec People Wiki: {word2vec_wiki.shape}")
    
    if glove_dict is not None:
        print(f"- GloVe Newsgroups: {glove_news.shape}")
        print(f"- GloVe People Wiki: {glove_wiki.shape}")