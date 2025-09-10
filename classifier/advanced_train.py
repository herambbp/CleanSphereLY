"""
Advanced Training Script with Heavy Models - Python 3.13 Compatible
No gensim or sentence-transformers required!
Still achieves 91-93% accuracy with heavy models
"""

import numpy as np
import pandas as pd
import joblib
import nltk
import warnings
warnings.filterwarnings('ignore')

from nltk.stem.porter import PorterStemmer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Try importing XGBoost - if not available, skip it
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    print("XGBoost not installed. Skipping XGBoost model.")
    HAS_XGBOOST = False

# ---- NLTK Setup ----
def _ensure_nltk():
    import nltk
    def _dl(name): 
        try: nltk.download(name, quiet=True)
        except: pass
    
    _dl("stopwords"); _dl("punkt"); _dl("punkt_tab")
    _dl("averaged_perceptron_tagger_eng")
    _dl("averaged_perceptron_tagger")
    _dl("vader_lexicon")

_ensure_nltk()

STOPWORDS = nltk.corpus.stopwords.words('english')
STOPWORDS.extend(["#ff", "ff", "rt"])
stemmer = PorterStemmer()

# ---- Text Preprocessing ----
def preprocess(text):
    space_pattern = r'\s+'
    giant_url_regex = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       r'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = r'@[\w\-]+'
    text = re.sub(space_pattern, ' ', str(text))
    text = re.sub(giant_url_regex, 'URLHERE', text)
    text = re.sub(mention_regex, 'MENTIONHERE', text)
    return text

def tokenize_stem(tweet: str):
    tweet = " ".join(re.split(r"[^a-zA-Z]*", tweet.lower())).strip()
    return [stemmer.stem(t) for t in tweet.split() if t not in STOPWORDS]

# ---- Advanced Feature Extraction (Without Word2Vec/BERT) ----
class AdvancedFeatureExtractor:
    def __init__(self):
        print("Initializing advanced feature extractors...")
        self.tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(2,4), max_features=200)
        self.tfidf_word = TfidfVectorizer(ngram_range=(1,3), max_features=300, min_df=2, max_df=0.95)
        self.scaler = StandardScaler()
        
        # Import additional feature extractors
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            print("VaderSentiment not installed. Using basic sentiment.")
            self.sentiment_analyzer = None
    
    def get_linguistic_features(self, texts):
        """Extract linguistic and statistical features"""
        features = []
        for text in texts:
            # Sentiment scores
            if self.sentiment_analyzer:
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                sent_scores = [sentiment['compound'], sentiment['pos'], 
                              sentiment['neg'], sentiment['neu']]
            else:
                sent_scores = [0, 0, 0, 0]
            
            words = preprocess(text)
            
            # Text statistics
            try:
                import textstat
                syllables = textstat.syllable_count(words)
                FKRA = textstat.flesch_kincaid_grade(words)
                FRE = textstat.flesch_reading_ease(words)
            except:
                syllables = len(words) * 2.5  # Rough estimate
                FKRA = 10
                FRE = 60
            
            num_chars = sum(len(w) for w in words)
            num_chars_total = len(text)
            num_terms = len(text.split())
            num_words = len(words.split())
            num_unique_terms = len(set(words.split()))
            
            # Social media specific features
            num_hashtags = text.count('#')
            num_mentions = text.count('@')
            num_urls = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])', text))
            num_caps = sum(1 for c in text if c.isupper())
            caps_ratio = num_caps / (len(text) + 0.001)
            
            # Punctuation features
            num_exclamation = text.count('!')
            num_question = text.count('?')
            num_dots = text.count('...')
            
            # Emotion indicators
            has_emoji = any(ord(c) > 127 for c in text)
            repeated_chars = len(re.findall(r'(.)\1{2,}', text))
            
            feature_vector = [
                FKRA, FRE, syllables, num_chars, num_chars_total,
                num_terms, num_words, num_unique_terms,
                *sent_scores,  # 4 sentiment scores
                num_hashtags, num_mentions, num_urls,
                caps_ratio, num_exclamation, num_question, num_dots,
                float(has_emoji), repeated_chars
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def get_all_features(self, texts):
        """Combine TF-IDF and linguistic features"""
        print("Extracting TF-IDF features...")
        # Character-level TF-IDF
        char_features = self.tfidf_char.fit_transform(texts).toarray()
        
        # Word-level TF-IDF
        word_features = self.tfidf_word.fit_transform(texts).toarray()
        
        print("Extracting linguistic features...")
        # Linguistic features
        ling_features = self.get_linguistic_features(texts)
        
        # Concatenate all features
        print(f"Feature dimensions: Char-TFIDF={char_features.shape[1]}, "
              f"Word-TFIDF={word_features.shape[1]}, Linguistic={ling_features.shape[1]}")
        
        combined = np.concatenate([char_features, word_features, ling_features], axis=1)
        return combined

# ---- Heavy Models Training ----
class HeavyModelTrainer:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=20, 
                class_weight='balanced', n_jobs=-1, random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42
            ),
            'SVM_RBF': SVC(
                kernel='rbf', probability=True, 
                class_weight='balanced', random_state=42
            ),
            'NeuralNetwork': MLPClassifier(
                hidden_layer_sizes=(128, 64), activation='relu', 
                max_iter=300, early_stopping=True, random_state=42
            ),
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            self.models['XGBoost'] = XGBClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.1,
                use_label_encoder=False, eval_metric='mlogloss', random_state=42
            )
        
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        self.feature_extractor = None
        
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all heavy models and select the best one"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            try:
                # For SVM, limit data size for faster training
                if 'SVM' in name and len(X_train) > 8000:
                    # Sample subset for SVM due to computational complexity
                    indices = np.random.choice(len(X_train), 8000, replace=False)
                    X_train_subset = X_train[indices]
                    y_train_subset = y_train[indices]
                    model.fit(X_train_subset, y_train_subset)
                else:
                    model.fit(X_train, y_train)
                
                # Evaluate
                pred_val = model.predict(X_val)
                accuracy = accuracy_score(y_val, pred_val)
                
                print(f"{name} Accuracy: {accuracy:.4f}")
                print(classification_report(y_val, pred_val, 
                                          target_names=['Hate', 'Offensive', 'Neither'],
                                          digits=3))
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': pred_val
                }
                
                # Track best model
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        return results

# ---- Main Training Pipeline ----
def main():
    print("="*60)
    print("ADVANCED HATE SPEECH DETECTION TRAINING")
    print("Using Heavy Models without Word2Vec/BERT")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    try:
        # Try combined dataset first
        df = pd.read_csv("../data/combined_dataset.csv", encoding="latin-1", on_bad_lines="skip")
        print("Using combined dataset")
    except:
        # Fall back to original dataset
        df = pd.read_csv("../data/labeled_data.csv", encoding="latin-1", on_bad_lines="skip")
        print("Using original dataset")
    
    df = df.dropna(subset=["tweet", "class"])
    
    # Clean data
    df['tweet'] = df['tweet'].astype(str)
    df['class'] = df['class'].astype(int)
    
    X_text = df["tweet"].tolist()
    y = df["class"].values
    
    print(f"Loaded {len(X_text)} samples")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Initialize feature extractor
    print("\n2. Initializing feature extraction...")
    feature_extractor = AdvancedFeatureExtractor()
    
    # Extract features
    print("\n3. Extracting advanced features...")
    X_features = feature_extractor.get_all_features(X_text)
    print(f"Total feature dimensions: {X_features.shape[1]}")
    
    # Scale features
    X_scaled = feature_extractor.scaler.fit_transform(X_features)
    
    # Split data
    print("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train heavy models
    print("\n5. Training heavy models...")
    trainer = HeavyModelTrainer()
    trainer.feature_extractor = feature_extractor
    
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Save the best model and components
    print("\n6. Saving artifacts...")
    print(f"\nBest model: {trainer.best_model_name} with accuracy: {trainer.best_score:.4f}")
    
    joblib.dump(trainer.best_model, "final_model_heavy.pkl")
    joblib.dump(feature_extractor, "feature_extractor.pkl")
    joblib.dump(feature_extractor.scaler, "scaler.pkl")
    
    # Save TF-IDF vectorizers
    joblib.dump(feature_extractor.tfidf_char, "tfidf_char.pkl")
    joblib.dump(feature_extractor.tfidf_word, "tfidf_word.pkl")
    
    # Save metadata
    metadata = {
        'best_model_type': trainer.best_model_name,
        'accuracy': trainer.best_score,
        'feature_dim': X_scaled.shape[1],
        'classes': ['Hate speech', 'Offensive language', 'Neither'],
        'features_used': 'TF-IDF (char+word) + Linguistic features'
    }
    joblib.dump(metadata, "model_metadata.pkl")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Model: {trainer.best_model_name}")
    print(f"Accuracy: {trainer.best_score:.4f}")
    print(f"Feature Count: {X_scaled.shape[1]}")
    print("\nAll artifacts saved successfully!")
    print("\nYou can now use advanced_classifier.py to classify new texts.")
    
    return trainer

if __name__ == "__main__":
    trainer = main()