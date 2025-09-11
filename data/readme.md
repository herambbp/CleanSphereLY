# Dataset Information

## Dataset Files

### Labeled Training Data
- **`labeled_data.csv`** (24,783 tweets): Primary crowdsourced training dataset
- **`labeled_data.p`**: Pickled pandas dataframe (Python 2.7 compatibility)

### Unlabeled Data
- **`trump_tweets_preprocessed.csv`** (29,804 tweets): Preprocessed Trump tweets
- **`personalities_tweets_preprocessed.csv`** (1,032 tweets): Preprocessed celebrity/personality tweets (labeled)
- **`reddit_comments_*.csv`** (~123K comments): Reddit discussions from news/politics subreddits
- **`combined_dataset.csv`** (105,619 entries): Combined labeled and unlabeled data from Twitter and Reddit sources

## Data Schema

Each dataset contains 6 columns:

**`count`** = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).

**`hate_speech`** = number of CF users who judged the tweet to be hate speech.

**`offensive_language`** = number of CF users who judged the tweet to be offensive.

**`neither`** = number of CF users who judged the tweet to be neither offensive nor non-offensive.

**`class`** = class label for majority of CF users.
  - 0 = hate speech
  - 1 = offensive language  
  - 2 = neither
  - -1 = unlabeled (for collected data)

**`tweet`** = preprocessed text content

## Preprocessing Applied

All datasets undergo consistent preprocessing:

### Basic Text Cleaning
- **URL Replacement**: HTTP links → "URLHERE"
- **Mention Normalization**: @username → "MENTIONHERE", /u/username → "MENTIONHERE"
- **Subreddit Links**: /r/subreddit → "SUBREDDITHERE" (Reddit only)
- **Whitespace Cleanup**: Multiple spaces collapsed to single space
- **Content Filtering**: Remove [deleted]/[removed] comments, minimum length 10+ characters

### Data Sources & Characteristics
- **Labeled Data**: Crowdsourced annotations from CrowdFlower platform
- **Trump Tweets**: Political discourse from 2016-2020 period (unlabeled)
- **Personalities**: 30 public figures across political leaders, business leaders, celebrities (labeled)
- **Reddit Comments**: News/politics discussion threads (unlabeled)

### Preprocessing Philosophy
- **Conservative cleaning**: Preserves authentic language patterns, emojis, slang
- **Platform standardization**: Consistent URL/mention handling across Twitter/Reddit
- **Minimal data loss**: Retains informal text structure and social media conventions
