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
    # Load balanced Reddit comments data
    input_file = 'raw/balanced_reddit_comments_20250911_193227.csv'
    print(f"Loading {input_file}...")
    
    try:
        reddit_df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    print(f"Loaded {len(reddit_df)} balanced Reddit comments")
    print(f"Columns: {list(reddit_df.columns)}")
    
    # Check the current data structure
    print(f"\nData structure check:")
    print(f"Unique subreddits: {reddit_df['subreddit'].nunique()}")
    print(f"Toxicity categories: {reddit_df['toxicity_category'].value_counts()}")
    print(f"Expected class distribution: {reddit_df['expected_class'].value_counts()}")
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
    
    # For balanced dataset, maintain the balanced nature by keeping all categories
    print(f"Maintaining balanced categories...")
    
    # Select only the required columns and rename for consistency
    # Use expected_class as the class since this is labeled balanced data
    processed_df = reddit_df[['count', 'hate_speech', 'offensive_language', 'neither', 'expected_class', 'tweet_processed']].copy()
    processed_df = processed_df.rename(columns={'tweet_processed': 'tweet', 'expected_class': 'class'})
    
    # Save the preprocessed balanced Reddit data
    output_file = 'balanced_reddit_comments_preprocessed.csv'
    processed_df.to_csv(output_file, index=False)
    print(f"\nPreprocessed balanced Reddit data saved to {output_file}")
    print(f"Final comment count: {len(processed_df)}")
    
    # Show sample of processed data
    print(f"\nFinal class distribution:")
    class_counts = processed_df['class'].value_counts().sort_index()
    class_names = {0: "Hate speech", 1: "Offensive language", 2: "Neither"}
    for class_val, count in class_counts.items():
        print(f"  {class_names.get(class_val, f'Class {class_val}')}: {count}")
    
    print("\nSample of processed comments:")
    for i, row in processed_df.head(3).iterrows():
        class_name = class_names.get(row['class'], f"Class {row['class']}")
        print(f"{i+1}. [{class_name}] {row['tweet'][:100]}...")
    
    # Load existing combined dataset and add balanced data
    print(f"\nLoading existing combined_dataset.csv...")
    try:
        combined_df = pd.read_csv('combined_dataset.csv')
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
        
        # Combine with balanced Reddit data
        print(f"\nCombining datasets...")
        updated_combined_df = pd.concat([combined_df, processed_df], ignore_index=True)
        
        # Save updated combined dataset
        updated_combined_df.to_csv('combined_dataset.csv', index=False)
        print(f"Updated combined dataset saved: {len(updated_combined_df)} total entries")
        
        # Show final composition
        print(f"\nFinal dataset composition:")
        print(f"  Previous data: {len(combined_df)}")
        print(f"  Balanced Reddit comments: {len(processed_df)}")
        print(f"  Total: {len(updated_combined_df)}")
        
        final_class_counts = updated_combined_df['class'].value_counts().sort_index()
        for class_val, count in final_class_counts.items():
            if class_val == -1:
                print(f"  Unlabeled: {count}")
            else:
                class_names = {0: "Hate speech", 1: "Offensive language", 2: "Neither"}
                print(f"  {class_names.get(class_val, f'Class {class_val}')}: {count}")
        
    except FileNotFoundError:
        print("combined_dataset.csv not found, creating new combined dataset with balanced data only")
        processed_df.to_csv('combined_dataset.csv', index=False)
    
    # Statistics
    print(f"\n=== Balanced Reddit Preprocessing Statistics ===")
    print(f"Average comment length: {processed_df['tweet'].str.len().mean():.1f} characters")
    print(f"Comments with URLHERE: {processed_df['tweet'].str.contains('URLHERE', na=False).sum()}")
    print(f"Comments with MENTIONHERE: {processed_df['tweet'].str.contains('MENTIONHERE', na=False).sum()}")
    print(f"Comments with SUBREDDITHERE: {processed_df['tweet'].str.contains('SUBREDDITHERE', na=False).sum()}")
    
    print(f"\nBalanced Reddit comments preprocessing completed!")

if __name__ == "__main__":
    main()