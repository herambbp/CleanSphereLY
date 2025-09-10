import pandas as pd
import re
import numpy as np

def additional_preprocessing(comment_text):
    """
    Apply additional preprocessing to Reddit comments
    The comments are already preprocessed but may need refinement
    """
    if pd.isna(comment_text):
        return ""
    
    # Convert to string
    comment_text = str(comment_text).strip()
    
    # Additional Reddit-specific cleaning
    # Remove common Reddit formatting artifacts
    comment_text = re.sub(r'^&gt;.*', '', comment_text, flags=re.MULTILINE)  # Remove quote lines
    comment_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', comment_text)  # Remove bold markdown
    comment_text = re.sub(r'\*([^*]+)\*', r'\1', comment_text)  # Remove italic markdown
    comment_text = re.sub(r'~~([^~]+)~~', r'\1', comment_text)  # Remove strikethrough
    
    # Clean up extra whitespace again after markdown removal
    comment_text = re.sub(r'\s+', ' ', comment_text.strip())
    
    return comment_text

def main():
    # Load Reddit comments data
    input_file = 'data/raw/reddit_comments_20250911_020222.csv'
    print(f"Loading {input_file}...")
    
    try:
        reddit_df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    print(f"Loaded {len(reddit_df)} Reddit comments")
    print(f"Columns: {list(reddit_df.columns)}")
    
    # Check the current data structure
    print(f"\nData structure check:")
    print(f"Unique subreddits: {reddit_df['subreddit'].nunique()}")
    print(f"Subreddits: {reddit_df['subreddit'].unique()}")
    print(f"Class distribution: {reddit_df['class'].value_counts()}")
    
    # Apply additional preprocessing to the comment text
    print(f"\nApplying additional preprocessing...")
    reddit_df['tweet_processed'] = reddit_df['comment_text'].apply(additional_preprocessing)
    
    # Content filtering
    original_count = len(reddit_df)
    
    # Remove very short comments (less than 15 characters after processing)
    reddit_df = reddit_df[reddit_df['tweet_processed'].str.len() >= 15].reset_index(drop=True)
    
    # Remove comments that are just common Reddit phrases or empty after processing
    common_phrases = ['[deleted]', '[removed]', 'This.', 'Same.', 'Agreed.', 'Yes.', 'No.', '^This', 'Exactly.']
    reddit_df = reddit_df[~reddit_df['tweet_processed'].isin(common_phrases)].reset_index(drop=True)
    
    filtered_count = len(reddit_df)
    print(f"Content filtering: {original_count} -> {filtered_count} comments (removed {original_count - filtered_count})")
    
    # Sample from large dataset for manageable size (optional - take first 50K comments)
    sample_size = min(50000, len(reddit_df))
    if len(reddit_df) > sample_size:
        reddit_df = reddit_df.head(sample_size).reset_index(drop=True)
        print(f"Sampled {sample_size} comments for processing")
    
    # Select only the required columns and rename for consistency
    processed_df = reddit_df[['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet_processed']].copy()
    processed_df = processed_df.rename(columns={'tweet_processed': 'tweet'})
    
    # Save the preprocessed Reddit data
    output_file = 'data/reddit_comments_preprocessed.csv'
    processed_df.to_csv(output_file, index=False)
    print(f"\nPreprocessed Reddit data saved to {output_file}")
    print(f"Final comment count: {len(processed_df)}")
    
    # Show sample of processed data
    print("\nSample of processed comments:")
    for i, row in processed_df.head(3).iterrows():
        print(f"{i+1}. {row['tweet'][:100]}...")
    
    # Load existing combined dataset
    print(f"\nLoading existing combined_dataset.csv...")
    try:
        combined_df = pd.read_csv('data/combined_dataset.csv')
        print(f"Existing combined dataset: {len(combined_df)} entries")
        
        # Check current composition
        print(f"Current dataset composition:")
        class_counts = combined_df['class'].value_counts().sort_index()
        for class_val, count in class_counts.items():
            if class_val == -1:
                print(f"  Unlabeled: {count}")
            else:
                class_names = {0: "Hate speech", 1: "Offensive language", 2: "Neither"}
                print(f"  {class_names.get(class_val, f'Class {class_val}')}: {count}")
        
        # Combine with Reddit data
        print(f"\nCombining datasets...")
        updated_combined_df = pd.concat([combined_df, processed_df], ignore_index=True)
        
        # Save updated combined dataset
        updated_combined_df.to_csv('data/combined_dataset.csv', index=False)
        print(f"Updated combined dataset saved: {len(updated_combined_df)} total entries")
        
        # Show final composition
        print(f"\nFinal dataset composition:")
        print(f"  Previous data: {len(combined_df)}")
        print(f"  Reddit comments: {len(processed_df)}")
        print(f"  Total: {len(updated_combined_df)}")
        
        final_class_counts = updated_combined_df['class'].value_counts().sort_index()
        for class_val, count in final_class_counts.items():
            if class_val == -1:
                print(f"  Unlabeled: {count}")
            else:
                class_names = {0: "Hate speech", 1: "Offensive language", 2: "Neither"}
                print(f"  {class_names.get(class_val, f'Class {class_val}')}: {count}")
        
    except FileNotFoundError:
        print("combined_dataset.csv not found")
        return
    
    # Statistics
    print(f"\n=== Reddit Preprocessing Statistics ===")
    print(f"Average comment length: {processed_df['tweet'].str.len().mean():.1f} characters")
    print(f"Comments with URLHERE: {processed_df['tweet'].str.contains('URLHERE', na=False).sum()}")
    print(f"Comments with MENTIONHERE: {processed_df['tweet'].str.contains('MENTIONHERE', na=False).sum()}")
    print(f"Comments with SUBREDDITHERE: {processed_df['tweet'].str.contains('SUBREDDITHERE', na=False).sum()}")
    
    print(f"\nReddit comments preprocessing completed!")

if __name__ == "__main__":
    main()