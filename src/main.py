import preprocessing as pre
import feature_extraction as FE
import clustering as clus
import evaluation 
import visualization as vis

def main():

    print("\n🚀 Starting Document Clustering Pipeline...\n")

    # # 1️⃣ Data Preprocessing
    # print("🔄 Step 1: Loading and preprocessing datasets...")
    # newsgroups_df = pre.load_newsgroups()  
    # people_wiki_df = pre.load_people_wiki()  
   
    # # Save data after cleaning
    # pre.save_cleaned_data(newsgroups_df,'cleaned_newsgroups.csv')
    # pre.save_cleaned_data(people_wiki_df,'cleaned_people_wiki.csv')  
    # print("✅ Preprocessing completed.\n")
    
    

    print("\n✅ Preprocessing completed and data saved!")


    # 2️⃣ Feature Extraction
    print("🔢 Step 2: Extracting features...")

    newsgroups_df, people_wiki_df = FE.load_cleaned_data()
    
    # TF-IDF application to texts
    vectorizer_news, tfidf_news = FE.extract_tfidf_features(newsgroups_df['cleaned_text'])
    vectorizer_wiki, tfidf_wiki = FE.extract_tfidf_features(people_wiki_df['cleaned_text'])
    
    # Word2Vec application to texts
    word2vec_model_news, word2vec_news = FE.extract_word2vec_features(newsgroups_df['cleaned_text'])
    word2vec_model_wiki, word2vec_wiki = FE.extract_word2vec_features(people_wiki_df['cleaned_text'])

    # Load the GloVe model
    glove_dict = FE.load_glove_model()

    # Extract GloVe features
    glove_news = FE.extract_glove_features(newsgroups_df['cleaned_text'], glove_dict)
    glove_wiki = FE.extract_glove_features(people_wiki_df['cleaned_text'], glove_dict)
    
    # Save TF-IDF models
    FE.save_model(vectorizer_news , 'models/tfidf_vectorizer_news.pkl')
    FE.save_model(vectorizer_wiki , 'models/tfidf_vectorizer_wiki.pkl')
    
    
    # Save TF-IDF matrices
    FE.save_matrix(tfidf_news.toarray() , 'data/vectorization_embedding_data/tfidf_newsgroups.csv')
    FE.save_matrix(tfidf_wiki.toarray() , 'data/vectorization_embedding_data/tfidf_people_wiki.csv')
    
    # Save Word2Vec forms and matrices
    word2vec_model_news.save('models/word2vec_news.model')
    word2vec_model_wiki.save('models/word2vec_wiki.model')
    FE.save_matrix(word2vec_news , 'data/vectorization_embedding_data/word2vec_newsgroups.csv')
    FE.save_matrix(word2vec_wiki , 'data/vectorization_embedding_data/word2vec_people_wiki.csv')

    # Save GloVe matrices
    FE.save_matrix(glove_news , 'data/vectorization_embedding_data/glove_newsgroups.csv')
    FE.save_matrix(glove_wiki , 'data/vectorization_embedding_data/glove_people_wiki.csv')

    print("✅ Feature extraction completed.\n")

    # 3️⃣ Clustering
    print("📊 Step 3: Applying clustering algorithms...")
    # load feature
    tfidf_news, tfidf_wiki, word2vec_news, word2vec_wiki, glove_news, glove_wiki = clus.load_features()

    # representations and algorithms
    representations = {
        "tfidf_news": tfidf_news, "tfidf_wiki": tfidf_wiki,
        "word2vec_news": word2vec_news, "word2vec_wiki": word2vec_wiki,
        "glove_news": glove_news, "glove_wiki": glove_wiki
    }

    methods = {
        "kmeans": clus.apply_kmeans,
        "hierarchical": clus.apply_hierarchical,
        "gmm": clus.apply_gmm,
        "lda": clus.apply_lda
    }

    # Storing classification results
    clustering_results = {}

    # Apply each algorithm to each representation.
    for rep_name, rep_data in representations.items():
        print(f"🔹 Applying clustering algorithms to {rep_name}...")

        for method_name, method_func in methods.items():
            model, labels = method_func(rep_data)  # Apply algorithm
            clustering_results[f"{method_name}_{rep_name}"] = labels  # save results
            
            print(f"✅ {method_name.upper()} applied to {rep_name}.")

    #save results
    for name, labels in clustering_results.items():
        clus.save_results(labels, name)


    print("✅ Clustering completed.\n")


    # 4️⃣ Evaluation
    print("🔍 Starting clustering evaluation...")
    
    evaluation.run_evaluation()

    print("✅ Evaluation completed.\n")

    # 5️⃣ Visualization
    print("🎨 Step 5: Generating visualizations...")
    vis.visualize_tsne_pca(method="kmeans", representation="tfidf_news")
    vis.visualize_tsne_pca(method="kmeans", representation="word2vec_news")
    
    vis.visualize_dendrogram(representation="tfidf_news")
    vis.visualize_dendrogram(representation="word2vec_news")
    print("✅ Visualization completed.\n")

    print("🎯 Clustering pipeline successfully completed! 🚀")

if __name__ == "__main__":
    main()
