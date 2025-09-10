import praw
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import time
import re

# Load environment variables
load_dotenv()

class RedditCommentScraper:
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
        """Apply basic preprocessing similar to tweet preprocessing"""
        if not comment_text or comment_text in ['[deleted]', '[removed]']:
            return None
            
        # Remove Reddit-specific formatting
        comment_text = re.sub(r'/u/\w+', 'MENTIONHERE', comment_text)  # User mentions
        comment_text = re.sub(r'/r/\w+', 'SUBREDDITHERE', comment_text)  # Subreddit mentions
        comment_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URLHERE', comment_text)
        
        # Clean whitespace
        comment_text = re.sub(r'\s+', ' ', comment_text.strip())
        
        return comment_text if len(comment_text) > 10 else None
    
    def scrape_subreddit_comments(self, subreddit_name, limit=1000, time_filter='week'):
        """
        Scrape comments from a specific subreddit
        
        Args:
            subreddit_name (str): Name of subreddit to scrape
            limit (int): Maximum number of posts to process
            time_filter (str): Time filter ('hour', 'day', 'week', 'month', 'year', 'all')
        """
        print(f"Scraping comments from r/{subreddit_name}...")
        
        subreddit = self.reddit.subreddit(subreddit_name)
        comments_data = []
        
        try:
            # Get hot posts from subreddit
            for post in subreddit.hot(limit=limit):
                print(f"Processing post: {post.title[:50]}...")
                
                # Expand all comments
                post.comments.replace_more(limit=0)
                
                # Extract comments
                for comment in post.comments.list():
                    if hasattr(comment, 'body'):
                        processed_text = self.preprocess_comment(comment.body)
                        
                        if processed_text:
                            comments_data.append({
                                'comment_id': comment.id,
                                'post_id': post.id,
                                'subreddit': subreddit_name,
                                'post_title': post.title,
                                'comment_text': processed_text,
                                'comment_score': comment.score,
                                'created_utc': datetime.fromtimestamp(comment.created_utc),
                                'author': str(comment.author) if comment.author else '[deleted]',
                                'is_submitter': comment.is_submitter,
                                'permalink': f"https://reddit.com{comment.permalink}"
                            })
                
                # Rate limiting
                time.sleep(0.1)
                
                if len(comments_data) % 100 == 0:
                    print(f"Collected {len(comments_data)} comments...")
        
        except Exception as e:
            print(f"Error scraping subreddit: {e}")
        
        return comments_data
    
    def scrape_multiple_subreddits(self, subreddit_list, limit_per_sub=500):
        """Scrape comments from multiple subreddits"""
        all_comments = []
        
        for subreddit_name in subreddit_list:
            try:
                comments = self.scrape_subreddit_comments(subreddit_name, limit=limit_per_sub)
                all_comments.extend(comments)
                print(f"Collected {len(comments)} comments from r/{subreddit_name}")
                time.sleep(1)  # Rate limiting between subreddits
            except Exception as e:
                print(f"Failed to scrape r/{subreddit_name}: {e}")
                continue
        
        return all_comments
    
    def save_to_csv(self, comments_data, filename=None):
        """Save comments data to CSV file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_comments_{timestamp}.csv"
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame(comments_data)
        
        # Add columns to match dataset format for potential training
        df['count'] = 0
        df['hate_speech'] = 0
        df['offensive_language'] = 0
        df['neither'] = 0
        df['class'] = -1  # Unlabeled
        df['tweet'] = df['comment_text']  # Alias for consistency
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} comments to {filename}")
        
        return filename, df

def main():
    """Example usage of Reddit comment scraper"""
    scraper = RedditCommentScraper()
    
    # Define subreddits to scrape (adjust based on your research needs)
    # Note: Choose subreddits carefully for hate speech research
    subreddits = [
        'news',
        'politics', 
        'worldnews',
        'unpopularopinion',
        'changemyview'
    ]
    
    print("Starting Reddit comment scraping...")
    
    # Scrape comments
    comments_data = scraper.scrape_multiple_subreddits(subreddits, limit_per_sub=200)
    
    if comments_data:
        # Save to CSV
        filename, df = scraper.save_to_csv(comments_data)
        
        # Display summary
        print(f"\nScraping Summary:")
        print(f"Total comments collected: {len(df)}")
        print(f"Average comment length: {df['comment_text'].str.len().mean():.1f} characters")
        print(f"Subreddits covered: {df['subreddit'].nunique()}")
        print(f"Unique posts: {df['post_id'].nunique()}")
        print(f"File saved: {filename}")
        
        # Show sample
        print(f"\nSample comments:")
        for i, row in df.head(3).iterrows():
            print(f"{i+1}. r/{row['subreddit']}: {row['comment_text'][:100]}...")
    else:
        print("No comments were collected.")

if __name__ == "__main__":
    main()