import praw
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import time
import re
import random

# Load environment variables
load_dotenv()

class BalancedRedditScraper:
    def __init__(self):
        """Initialize Reddit API connection using credentials from .env"""
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        
        # Test connection
        try:
            print(f"Connected to Reddit API as read-only: {self.reddit.read_only}")
        except Exception as e:
            print(f"Error connecting to Reddit API: {e}")
            raise
    
    def preprocess_comment(self, comment_text):
        """Apply preprocessing similar to existing pipeline"""
        if not comment_text or comment_text in ['[deleted]', '[removed]']:
            return None
            
        # Remove Reddit-specific formatting
        comment_text = re.sub(r'/u/\w+', 'MENTIONHERE', comment_text)  # User mentions
        comment_text = re.sub(r'/r/\w+', 'SUBREDDITHERE', comment_text)  # Subreddit mentions
        comment_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URLHERE', comment_text)
        
        # Remove Reddit markdown formatting
        comment_text = re.sub(r'^&gt;.*', '', comment_text, flags=re.MULTILINE)  # Remove quote lines
        comment_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', comment_text)  # Remove bold
        comment_text = re.sub(r'\*([^*]+)\*', r'\1', comment_text)  # Remove italic
        comment_text = re.sub(r'~~([^~]+)~~', r'\1', comment_text)  # Remove strikethrough
        
        # Clean whitespace
        comment_text = re.sub(r'\s+', ' ', comment_text.strip())
        
        return comment_text if len(comment_text) > 15 else None
    
    def get_balanced_subreddits(self):
        """
        Define balanced subreddit collection strategy
        Returns dict with toxicity categories and corresponding subreddits
        """
        return {
            # Low toxicity - Normal discourse (60% of collection)
            'low_toxicity': {
                'subreddits': [
                    'explainlikeimfive',  # Educational content
                    'todayilearned',      # Factual sharing
                    'science',            # Scientific discussion
                    'books',              # Literature discussion
                    'cooking',            # Hobby discussion
                    'mildlyinteresting'   # Casual observations
                ],
                'limit_per_sub': 300,
                'expected_class': 2    # Neither hate nor offensive
            },
            
            # Medium toxicity - Opinion/debate (30% of collection)
            'medium_toxicity': {
                'subreddits': [
                    'unpopularopinion',   # Controversial opinions
                    'AmItheAsshole',      # Judgment discussions
                    'TrueOffMyChest',     # Venting content
                    'changemyview',       # Debate content
                    'mildlyinfuriating'   # Frustration expression
                ],
                'limit_per_sub': 200,
                'expected_class': 1    # Potentially offensive language
            },
            
            # Higher risk - Conflict content (10% of collection)
            'high_risk': {
                'subreddits': [
                    'SubredditDrama',     # Conflict discussion
                    'PublicFreakout',     # Heated situations
                    'trashy'              # Judgmental content
                ],
                'limit_per_sub': 100,
                'expected_class': 0    # Potential hate speech
            }
        }
    
    def scrape_balanced_subreddits(self):
        """Scrape comments from balanced subreddit collection"""
        subreddit_config = self.get_balanced_subreddits()
        all_comments = []
        
        for toxicity_level, config in subreddit_config.items():
            print(f"\n=== Collecting {toxicity_level.upper()} content ===")
            
            for subreddit_name in config['subreddits']:
                try:
                    print(f"Scraping r/{subreddit_name} (target: {config['limit_per_sub']} posts)...")
                    
                    subreddit = self.reddit.subreddit(subreddit_name)
                    post_count = 0
                    comment_count = 0
                    
                    # Get hot posts from subreddit
                    for post in subreddit.hot(limit=config['limit_per_sub']):
                        post_count += 1
                        
                        # Expand comments (limit to avoid overwhelming)
                        post.comments.replace_more(limit=2)
                        
                        # Extract comments
                        for comment in post.comments.list()[:10]:  # Limit comments per post
                            if hasattr(comment, 'body'):
                                processed_text = self.preprocess_comment(comment.body)
                                
                                if processed_text:
                                    all_comments.append({
                                        'comment_id': comment.id,
                                        'post_id': post.id,
                                        'subreddit': subreddit_name,
                                        'toxicity_category': toxicity_level,
                                        'expected_class': config['expected_class'],
                                        'post_title': post.title,
                                        'comment_text': processed_text,
                                        'comment_score': comment.score,
                                        'created_utc': datetime.fromtimestamp(comment.created_utc),
                                        'author': str(comment.author) if comment.author else '[deleted]',
                                        'permalink': f"https://reddit.com{comment.permalink}"
                                    })
                                    comment_count += 1
                        
                        # Rate limiting
                        time.sleep(0.1)
                        
                        if post_count % 50 == 0:
                            print(f"  Processed {post_count} posts, collected {comment_count} comments...")
                    
                    print(f"  Completed r/{subreddit_name}: {comment_count} comments from {post_count} posts")
                    time.sleep(1)  # Rate limiting between subreddits
                    
                except Exception as e:
                    print(f"  Error scraping r/{subreddit_name}: {e}")
                    continue
        
        return all_comments
    
    def save_balanced_dataset(self, comments_data, filename=None):
        """Save balanced comments data to CSV with analysis"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"raw/balanced_reddit_comments_{timestamp}.csv"
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame(comments_data)
        
        if df.empty:
            print("No data collected!")
            return None, None
        
        # Add dataset schema columns
        df['count'] = 0
        df['hate_speech'] = 0
        df['offensive_language'] = 0
        df['neither'] = 0
        df['class'] = -1  # Initially unlabeled
        df['tweet'] = df['comment_text']  # Alias for consistency
        
        # Analyze balance
        print(f"\n=== Collection Balance Analysis ===")
        toxicity_dist = df['toxicity_category'].value_counts()
        total_comments = len(df)
        
        for category, count in toxicity_dist.items():
            percentage = (count / total_comments) * 100
            print(f"{category}: {count} comments ({percentage:.1f}%)")
        
        subreddit_dist = df.groupby(['toxicity_category', 'subreddit']).size().reset_index(name='count')
        print(f"\nSubreddit distribution:")
        for _, row in subreddit_dist.iterrows():
            print(f"  {row['toxicity_category']}: r/{row['subreddit']} = {row['count']} comments")
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"\nSaved {len(df)} balanced comments to {filename}")
        
        return filename, df
    
    def create_sample_labels(self, df):
        """
        Create sample labels based on toxicity categories for demonstration
        In practice, these would need manual annotation
        """
        print(f"\n=== Creating Sample Labels (for demonstration) ===")
        
        # This is just for demonstration - real labels need manual annotation
        for idx, row in df.iterrows():
            if row['toxicity_category'] == 'low_toxicity':
                # Assume most low toxicity comments are "neither"
                df.at[idx, 'class'] = 2
                df.at[idx, 'neither'] = 3
            elif row['toxicity_category'] == 'medium_toxicity':
                # Mix of offensive and neither
                df.at[idx, 'class'] = random.choice([1, 2])
                if df.at[idx, 'class'] == 1:
                    df.at[idx, 'offensive_language'] = 3
                else:
                    df.at[idx, 'neither'] = 3
            elif row['toxicity_category'] == 'high_risk':
                # Mix of hate speech, offensive, and neither
                df.at[idx, 'class'] = random.choice([0, 1, 2])
                if df.at[idx, 'class'] == 0:
                    df.at[idx, 'hate_speech'] = 3
                elif df.at[idx, 'class'] == 1:
                    df.at[idx, 'offensive_language'] = 3
                else:
                    df.at[idx, 'neither'] = 3
            
            df.at[idx, 'count'] = 3  # Simulated annotator count
        
        print("WARNING: These are simulated labels for demonstration!")
        print("For real research, manual annotation by trained annotators is required.")
        
        return df

def main():
    """Run balanced Reddit comment collection"""
    scraper = BalancedRedditScraper()
    
    print("=== Balanced Reddit Comment Collection ===")
    print("Strategy: Collect from subreddits with different toxicity levels")
    print("- Low toxicity (60%): Educational, factual, hobby content")
    print("- Medium toxicity (30%): Opinion, debate, venting content")  
    print("- High risk (10%): Conflict, drama, judgmental content")
    
    # Get user confirmation
    proceed = input("\nProceed with balanced collection? (y/n): ").lower().strip()
    if proceed != 'y':
        print("Collection cancelled.")
        return
    
    # Scrape balanced comments
    print("\nStarting balanced Reddit comment collection...")
    comments_data = scraper.scrape_balanced_subreddits()
    
    if comments_data:
        # Save dataset
        filename, df = scraper.save_balanced_dataset(comments_data)
        
        if df is not None:
            # Optional: Create sample labels
            create_labels = input("\nCreate sample labels for demonstration? (y/n): ").lower().strip()
            if create_labels == 'y':
                df = scraper.create_sample_labels(df)
                
                # Save labeled version
                labeled_filename = filename.replace('.csv', '_with_sample_labels.csv')
                df.to_csv(labeled_filename, index=False)
                print(f"Sample labeled dataset saved to {labeled_filename}")
                
                # Show class distribution
                print(f"\nSample label distribution:")
                class_dist = df['class'].value_counts().sort_index()
                class_names = {-1: "Unlabeled", 0: "Hate speech", 1: "Offensive", 2: "Neither"}
                for class_val, count in class_dist.items():
                    print(f"  {class_names.get(class_val, f'Class {class_val}')}: {count}")
            
            # Show samples
            print(f"\nSample comments by category:")
            for category in df['toxicity_category'].unique():
                category_samples = df[df['toxicity_category'] == category].head(2)
                print(f"\n{category.upper()}:")
                for _, sample in category_samples.iterrows():
                    print(f"  r/{sample['subreddit']}: {sample['comment_text'][:80]}...")
    else:
        print("No comments were collected.")

if __name__ == "__main__":
    main()