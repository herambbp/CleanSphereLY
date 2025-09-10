"""
Advanced Training Script with Heavy Models and Better Text Representations
This replaces the simple LogisticRegression with multiple heavy models
and upgrades from TF-IDF to advanced embeddings
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
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import re

# For advanced text representations
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import torch

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

def basic_tokenize(tweet: str):
    tweet = " ".join(re.split(r"[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

# ---- Advanced Feature Extraction ----
class AdvancedFeatureExtractor:
    def __init__(self):
        print("Initializing advanced feature extractors...")
        # Initialize Sentence-BERT for embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.word2vec_model = None
        self.scaler = StandardScaler()
        
        # Import additional feature extractors
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        import textstat
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def train_word2vec(self, texts):
        """Train Word2Vec on the corpus"""
        print("Training Word2Vec model...")
        tokenized_texts = [tokenize_stem(text) for text in texts]
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            epochs=10
        )
        
    def get_word2vec_features(self, texts):
        """Convert texts to Word2Vec embeddings (average of word vectors)"""
        features = []
        for text in texts:
            tokens = tokenize_stem(text)
            vectors = []
            for token in tokens:
                if token in self.word2vec_model.wv:
                    vectors.append(self.word2vec_model.wv[token])
            if vectors:
                features.append(np.mean(vectors, axis=0))
            else:
                features.append(np.zeros(100))
        return np.array(features)
    
    def get_bert_embeddings(self, texts):
        """Get BERT-based sentence embeddings"""
        print("Generating BERT embeddings...")
        embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def get_linguistic_features(self, texts):
        """Extract linguistic and statistical features"""
        features = []
        for text in texts:
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            words = preprocess(text)
            
            # Text statistics
            import textstat
            syllables = textstat.syllable_count(words)
            num_chars = sum(len(w) for w in words)
            num_chars_total = len(text)
            num_terms = len(text.split())
            num_words = len(words.split())
            avg_syl = (syllables + 0.001) / (num_words + 0.001)
            num_unique_terms = len(set(words.split()))
            
            # Readability scores
            FKRA = 0.39*(num_words/1.0) + 11.8*avg_syl - 15.59
            FRE = 206.835 - 1.015*(num_words/1.0) - (84.6*avg_syl)
            
            # Social media specific features
            num_hashtags = text.count('#')
            num_mentions = text.count('@')
            num_urls = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])', text))
            num_caps = sum(1 for c in text if c.isupper())
            caps_ratio = num_caps / (len(text) + 0.001)
            
            # Punctuation features
            num_exclamation = text.count('!')
            num_question = text.count('?')
            
            feature_vector = [
                FKRA, FRE, syllables, num_chars, num_chars_total,
                num_terms, num_words, num_unique_terms,
                sentiment['compound'], sentiment['pos'], sentiment['neg'], sentiment['neu'],
                num_hashtags, num_mentions, num_urls,
                caps_ratio, num_exclamation, num_question
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def get_all_features(self, texts, use_word2vec=True, use_bert=True):
        """Combine all feature types"""
        all_features = []
        
        # Get linguistic features
        ling_features = self.get_linguistic_features(texts)
        all_features.append(ling_features)
        
        # Get Word2Vec features
        if use_word2vec and self.word2vec_model:
            w2v_features = self.get_word2vec_features(texts)
            all_features.append(w2v_features)
        
        # Get BERT embeddings
        if use_bert:
            bert_features = self.get_bert_embeddings(texts)
            all_features.append(bert_features)
        
        # Concatenate all features
        combined = np.concatenate(all_features, axis=1)
        return combined

# ---- Heavy Models Training ----
class HeavyModelTrainer:
    def __init__(self):
        self.models = {
            'SVM_RBF': SVC(kernel='rbf', probability=True, class_weight='balanced'),
            'SVM_Poly': SVC(kernel='poly', degree=3, probability=True, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=20, 
                                                  class_weight='balanced', n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1,
                                    use_label_encoder=False, eval_metric='mlogloss'),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=150, max_depth=10),
            'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(256, 128, 64), 
                                          activation='relu', max_iter=500, early_stopping=True)
        }
        self.best_model = None
        self.best_score = 0
        self.feature_extractor = None
        
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all heavy models and select the best one"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            try:
                # For SVM, limit data size for faster training
                if 'SVM' in name and len(X_train) > 10000:
                    # Sample subset for SVM due to computational complexity
                    indices = np.random.choice(len(X_train), 10000, replace=False)
                    X_train_subset = X_train[indices]
                    y_train_subset = y_train[indices]
                    model.fit(X_train_subset, y_train_subset)
                else:
                    model.fit(X_train, y_train)
                
                # Evaluate
                pred_val = model.predict(X_val)
                accuracy = accuracy_score(y_val, pred_val)
                
                print(f"{name} Accuracy: {accuracy:.4f}")
                print(classification_report(y_val, pred_val, target_names=['Hate', 'Offensive', 'Neither']))
                
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
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='XGBoost'):
        """Perform hyperparameter tuning for the best model"""
        print(f"\nPerforming hyperparameter tuning for {model_type}...")
        
        if model_type == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            }
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        elif model_type == 'RandomForest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(class_weight='balanced')
        else:
            return None
        
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_

# ---- Main Training Pipeline ----
def main():
    print("="*60)
    print("ADVANCED HATE SPEECH DETECTION TRAINING")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv("../data/labeled_data.csv", encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["tweet", "class"])
    
    X_text = df["tweet"].astype(str).tolist()
    y = df["class"].astype(int).values
    
    print(f"Loaded {len(X_text)} samples")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Initialize feature extractor
    print("\n2. Initializing advanced feature extraction...")
    feature_extractor = AdvancedFeatureExtractor()
    
    # Train Word2Vec on the corpus
    feature_extractor.train_word2vec(X_text)
    
    # Extract features
    print("\n3. Extracting advanced features...")
    X_features = feature_extractor.get_all_features(X_text, use_word2vec=True, use_bert=True)
    
    # Scale features
    X_scaled = feature_extractor.scaler.fit_transform(X_features)
    
    # Split data
    print("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train heavy models
    print("\n5. Training heavy models...")
    trainer = HeavyModelTrainer()
    trainer.feature_extractor = feature_extractor
    
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Hyperparameter tuning on best model
    print("\n6. Hyperparameter tuning on best model...")
    if trainer.best_model_name in ['XGBoost', 'RandomForest']:
        tuned_model = trainer.hyperparameter_tuning(X_train, y_train, trainer.best_model_name)
        if tuned_model:
            pred_test = tuned_model.predict(X_test)
            tuned_accuracy = accuracy_score(y_test, pred_test)
            print(f"\nTuned model accuracy: {tuned_accuracy:.4f}")
            
            if tuned_accuracy > trainer.best_score:
                trainer.best_model = tuned_model
                trainer.best_score = tuned_accuracy
    
    # Save the best model and components
    print("\n7. Saving artifacts...")
    print(f"Best model: {trainer.best_model_name} with accuracy: {trainer.best_score:.4f}")
    
    joblib.dump(trainer.best_model, "final_model_heavy.pkl")
    joblib.dump(feature_extractor, "feature_extractor.pkl")
    joblib.dump(feature_extractor.scaler, "scaler.pkl")
    joblib.dump(feature_extractor.word2vec_model, "word2vec_model.pkl")
    
    # Save metadata
    metadata = {
        'best_model_type': trainer.best_model_name,
        'accuracy': trainer.best_score,
        'feature_dim': X_scaled.shape[1],
        'classes': ['Hate speech', 'Offensive language', 'Neither']
    }
    joblib.dump(metadata, "model_metadata.pkl")
    
    print("\nTraining complete! All artifacts saved.")
    print(f"Final model accuracy: {trainer.best_score:.4f}")
    
    return trainer

if __name__ == "__main__":
    trainer = main()