import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_tweet(tweet):
    """
    Apply basic preprocessing following CleanSphereLY pipeline
    Conservative approach - preserves authentic language patterns
    """
    if pd.isna(tweet):
        return ""
    
    # Convert to string and remove extra whitespace
    tweet = str(tweet).strip()
    
    # Replace URLs with URLHERE (Conservative - keeps placeholder for context)
    tweet = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URLHERE', tweet)
    
    # Replace mentions with MENTIONHERE (Standardize across platforms)
    tweet = re.sub(r'@[A-Za-z0-9_]+', 'MENTIONHERE', tweet)
    
    # Clean whitespace (Collapse multiple spaces to single space)
    tweet = re.sub(r'\s+', ' ', tweet)
    
    return tweet.strip()

def tokenize_and_stem(tweet):
    """
    Advanced processing for training pipeline
    This happens during model training, not data preprocessing
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Remove punctuation and convert to lowercase (Training-time processing)
    translator = str.maketrans('', '', string.punctuation)
    tweet = tweet.translate(translator).lower()
    
    # Split into tokens (keeping only alphabetic characters)
    tokens = re.findall(r'[a-zA-Z]+', tweet)
    
    # Remove stopwords and stem
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(stemmed_tokens)

def main():
    # Load trump tweets data from raw directory
    input_file = 'data/raw/trump_tweets.csv'
    print(f"Loading {input_file}...")
    
    # Handle different encodings
    try:
        trump_df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            trump_df = pd.read_csv(input_file, encoding='latin1')
        except UnicodeDecodeError:
            trump_df = pd.read_csv(input_file, encoding='cp1252')
    
    print(f"Loaded {len(trump_df)} raw tweets")
    print(f"Columns: {list(trump_df.columns)}")
    
    # Check for Text column (main content)
    if 'Text' not in trump_df.columns:
        print("Error: 'Text' column not found in trump_tweets.csv")
        print(f"Available columns: {list(trump_df.columns)}")
        return
    
    # Clean and preprocess the text column
    print(f"Preprocessing {len(trump_df)} tweets...")
    trump_df['tweet'] = trump_df['Text'].apply(preprocess_tweet)
    
    # Content filtering - remove empty tweets
    original_count = len(trump_df)
    trump_df = trump_df[trump_df['tweet'].str.len() > 10].reset_index(drop=True)
    filtered_count = len(trump_df)
    
    print(f"Content filtering: {original_count} → {filtered_count} tweets (removed {original_count - filtered_count} short tweets)")
    
    # Add dataset schema columns to match existing format
    trump_df['count'] = 0  # No crowd annotations
    trump_df['hate_speech'] = 0
    trump_df['offensive_language'] = 0
    trump_df['neither'] = 0
    trump_df['class'] = -1  # Unlabeled data marker
    
    # Select only the columns that match labeled_data.csv format
    processed_df = trump_df[['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet']].copy()
    
    # Save the preprocessed data
    output_file = 'data/trump_tweets_preprocessed.csv'
    processed_df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")
    print(f"Final tweet count: {len(processed_df)}")
    
    # Show sample of processed data
    print("\\nSample of processed tweets:")
    for i, row in processed_df.head(3).iterrows():
        original_text = trump_df.iloc[i]['Text'][:100] + "..." if len(trump_df.iloc[i]['Text']) > 100 else trump_df.iloc[i]['Text']
        processed_text = row['tweet'][:100] + "..." if len(row['tweet']) > 100 else row['tweet']
        print(f"\\nOriginal:  {original_text}")
        print(f"Processed: {processed_text}")
    
    # Statistics
    print("\\n=== Preprocessing Statistics ===")
    print(f"Average tweet length: {processed_df['tweet'].str.len().mean():.1f} characters")
    print(f"Tweets with URLHERE: {processed_df['tweet'].str.contains('URLHERE').sum()}")
    print(f"Tweets with MENTIONHERE: {processed_df['tweet'].str.contains('MENTIONHERE').sum()}")
    
    # Optional: Update combined dataset
    update_combined = input("\\nUpdate combined_dataset.csv with new Trump data? (y/n): ").lower().strip()
    
    if update_combined == 'y':
        print("\\nUpdating combined dataset...")
        
        # Load existing combined dataset
        try:
            combined_df = pd.read_csv('data/combined_dataset.csv')
            print(f"Loaded existing combined dataset: {len(combined_df)} entries")
            
            # Remove old Trump data (class = -1) 
            labeled_data = combined_df[combined_df['class'] != -1]
            print(f"Keeping labeled data: {len(labeled_data)} entries")
            
            # Add new Trump data
            updated_combined_df = pd.concat([labeled_data, processed_df], ignore_index=True)
            
            # Save updated combined dataset
            updated_combined_df.to_csv('data/combined_dataset.csv', index=False)
            print(f"Updated combined dataset saved: {len(updated_combined_df)} total entries")
            print(f"  - Labeled: {len(labeled_data)}")
            print(f"  - Trump tweets: {len(processed_df)}")
            
        except FileNotFoundError:
            print("combined_dataset.csv not found, skipping update")
    
    print("\\nTrump tweets preprocessing completed!")

if __name__ == "__main__":
    main()