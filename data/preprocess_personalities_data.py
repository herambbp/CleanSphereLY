import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

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
    """Preprocess tweet text following the same pipeline as the existing classifier"""
    if pd.isna(tweet):
        return ""
    
    # Convert to string and remove extra whitespace
    tweet = str(tweet).strip()
    
    # Replace URLs with URLHERE
    tweet = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URLHERE', tweet)
    
    # Replace mentions with MENTIONHERE
    tweet = re.sub(r'@[A-Za-z0-9_]+', 'MENTIONHERE', tweet)
    
    # Remove extra whitespace
    tweet = re.sub(r'\s+', ' ', tweet)
    
    return tweet.strip()

def main():
    # Load personalities tweets data
    print("Loading personalities_tweets.csv...")
    personalities_df = pd.read_csv('data/personalities_tweets.csv')
    
    print(f"Original dataset: {len(personalities_df)} tweets")
    print(f"Personalities included: {personalities_df['personality'].nunique()}")
    print(f"Categories: {personalities_df['category'].unique()}")
    print(f"Class distribution: {personalities_df['class'].value_counts().to_dict()}")
    
    # Clean and preprocess the text column
    print(f"Processing {len(personalities_df)} tweets...")
    personalities_df['tweet_processed'] = personalities_df['tweet'].apply(preprocess_tweet)
    
    # Remove empty tweets
    personalities_df = personalities_df[personalities_df['tweet_processed'].str.len() > 0].reset_index(drop=True)
    
    # Map the classification scheme to match existing dataset format
    # Current classes: 1 (offensive), 2 (neither) - need to map to proper scheme
    print("\\nOriginal class distribution:")
    print(personalities_df['class'].value_counts())
    
    # Since these tweets are from personalities and already labeled, 
    # we'll keep their existing labels but need to add missing annotation columns
    personalities_df['count'] = 3  # Assume equivalent to 3 annotators for synthetic data
    personalities_df['hate_speech'] = 0
    personalities_df['offensive_language'] = 0  
    personalities_df['neither'] = 0
    
    # Map the class labels and set annotation counts accordingly
    for idx, row in personalities_df.iterrows():
        if row['class'] == 0:  # hate speech
            personalities_df.at[idx, 'hate_speech'] = 3
        elif row['class'] == 1:  # offensive language  
            personalities_df.at[idx, 'offensive_language'] = 3
        elif row['class'] == 2:  # neither
            personalities_df.at[idx, 'neither'] = 3
    
    # Select columns to match combined_dataset.csv format
    processed_df = personalities_df[['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet_processed']].copy()
    processed_df = processed_df.rename(columns={'tweet_processed': 'tweet'})
    
    # Save the preprocessed data
    output_file = 'data/personalities_tweets_preprocessed.csv'
    processed_df.to_csv(output_file, index=False)
    print(f"\\nPreprocessed data saved to {output_file}")
    print(f"Total tweets processed: {len(processed_df)}")
    
    # Show sample of processed data
    print("\\nSample of processed tweets:")
    print(processed_df.head())
    
    # Load existing combined dataset
    print("\\nLoading existing combined_dataset.csv...")
    combined_df = pd.read_csv('data/combined_dataset.csv')
    print(f"Existing combined dataset: {len(combined_df)} entries")
    
    # Combine with personalities data
    updated_combined_df = pd.concat([combined_df, processed_df], ignore_index=True)
    
    # Save updated combined dataset
    updated_output = 'data/combined_dataset.csv'
    updated_combined_df.to_csv(updated_output, index=False)
    print(f"\\nUpdated combined dataset saved to {updated_output}")
    print(f"Total entries: {len(updated_combined_df)}")
    
    # Show distribution summary
    print(f"\\nFinal dataset composition:")
    print(f"- Labeled tweets (original): {len(combined_df[combined_df['class'] != -1])}")
    print(f"- Unlabeled tweets (Trump): {len(combined_df[combined_df['class'] == -1])}")  
    print(f"- Personalities tweets: {len(processed_df)}")
    print(f"- Total: {len(updated_combined_df)}")
    
    print(f"\\nClass distribution in final dataset:")
    class_counts = updated_combined_df['class'].value_counts().sort_index()
    for class_val, count in class_counts.items():
        if class_val == -1:
            print(f"  Unlabeled: {count}")
        else:
            class_names = {0: "Hate speech", 1: "Offensive language", 2: "Neither"}
            print(f"  {class_names.get(class_val, f'Class {class_val}')}: {count}")

if __name__ == "__main__":
    main()