import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import re
import spacy
from nltk.corpus import stopwords
import os

# Load the English language form from spaCy
nlp = spacy.load("en_core_web_sm")

# Load the list of common words (Stopwords)
stop_words = set(stopwords.words('english'))



def clean_text(text):
    """
    Text cleaning: remove symbols, numbers, and convert text to lowercase.
    """
    if pd.isna(text) or text == "":
        return ""  # Return empty string for NaN or empty values
        
    text = str(text).lower()  # Convert to lowercase and ensure it's a string
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove special symbols and tags
    text = text.strip()  # Remove extra spaces
    return text


def preprocess_text(text):
    """
    A text cleaning application that removes common words and returns words to their original form (lemmatization).
    """
    if pd.isna(text) or text == "":
        return ""  # Return empty string for NaN or empty values
        
    text = clean_text(text)  # Text cleaning
    
    # Skip processing if text is empty after cleaning
    if not text:
        return ""
        
    doc = nlp(text)  # Text analysis using spaCy
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return ' '.join(tokens)


# 🔹 Read data from 20 Newsgroups
def load_newsgroups(categories=['talk.religion.misc', 'comp.graphics', 'sci.space']):
    """
    Download and process data from 20 Newsgroups and save true labels.
    """
    print("📥 Loading 20 Newsgroups dataset...")
    
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    df_raw_texts = pd.DataFrame({'raw_text': newsgroups.data})
    
    df_raw_texts.to_csv("data/Dataset/raw_newsgroups.csv", index=False)
    print("📂 Raw Newsgroups dataset saved before preprocessing!")

    # Apply preprocessing and track documents with issues
    processed_texts = []
    empty_after_processing = 0
    empty_before_processing = 0
    
    for text in newsgroups.data:
        if not text:
            empty_before_processing += 1
        processed = preprocess_text(text)
        processed_texts.append(processed)
        if not processed:
            empty_after_processing += 1
    
    if empty_before_processing > 0:
        print(f"⚠️ Warning: {empty_before_processing} documents is empty befor preprocessing")
    

    if empty_after_processing > 0:
        print(f"⚠️ Warning: {empty_after_processing} documents became empty after preprocessing")
    
    df_texts = pd.DataFrame({'cleaned_text': processed_texts})


    # save Real Categories
    df_labels = pd.DataFrame({'category': newsgroups.target})
    df_labels.to_csv('data/Real_Categories/true_labels_newsgroups.csv', index=False)

    print("✅ Loading 20 Newsgroups dataset & Save it's Real Categories is done!")
    return df_texts

# 🔹 Read People Wikipedia data
def load_people_wiki():
    """
    load and process Wikipedia data
    """
    print("📥 Loading People Wikipedia dataset...")
    
    try:
        df = pd.read_csv('data/Dataset/people_wiki.csv')
        
        # Check for missing data in raw dataset
        empty_before_processing = (df['text'] == "").sum()
        if empty_before_processing > 0:
            print(f"⚠️ Warning: {empty_before_processing} documents is empty before preprocessing")
      

        # Apply preprocessing
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        
        # Check for missing or empty data after preprocessing
        empty_after_processing = (df['cleaned_text'] == "").sum()
        if empty_after_processing > 0:
            print(f"⚠️ Warning: {empty_after_processing} documents became empty after preprocessing")
        
        # Create a new dataframe with just the columns we need
        result_df = df[['name', 'cleaned_text']]
        
        # Check for missing data after preprocessing        
        print("✅ Loading People Wikipedia dataset is done...")
        return result_df
        
    except FileNotFoundError:
        print("❌ Error: people_wiki.csv file not found in data/Dataset/ directory")
        print("Please ensure the file exists at the specified location")
        return pd.DataFrame(columns=['name', 'cleaned_text'])


def save_cleaned_data(df, name):
    """
    This function saves the cleaned data in CSV files for later use.
    """
    # Check if dataframes are empty before saving
    if df.empty:
        print("⚠️ Warning: dataframe is empty, not saving")
    else:
        df.to_csv(f"data/Preprocessed_Cleand_Data/{name}", index=False)
        print(f"✅ {name} cleaned data saved successfully")


if __name__ == "__main__":
    print("🔍 Starting data preprocessing pipeline...")
    
    # Process newsgroups data
    newsgroups_df = load_newsgroups()
    print("\n📝 Newsgroups data sample:")
    print(newsgroups_df.head())
    
    # Process people wiki data
    people_wiki_df = load_people_wiki()
    print("\n📝 People Wikipedia data sample:")
    print(people_wiki_df.head())

    # Save Cleaned Data into CSV Files 
    save_cleaned_data(newsgroups_df,'cleaned_newsgroups.csv') 
    save_cleaned_data(people_wiki_df,'cleaned_people_wiki.csv') 

    print("\n✅ Preprocessing completed and data saved!")