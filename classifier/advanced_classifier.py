"""
Advanced Classifier - Works with the trained model files
Must have the same class definitions as used during training
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
import warnings
import re
import nltk
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.stem.porter import PorterStemmer

# Ensure NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOPWORDS = nltk.corpus.stopwords.words('english')
STOPWORDS.extend(["#ff", "ff", "rt"])
stemmer = PorterStemmer()

# ---- MUST MATCH THE TRAINING SCRIPT EXACTLY ----
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

# This class MUST match the one used during training
class AdvancedFeatureExtractor:
    def __init__(self):
        print("Initializing advanced feature extractors...")
        self.tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(2,4), max_features=200)
        self.tfidf_word = TfidfVectorizer(ngram_range=(1,3), max_features=300, min_df=2, max_df=0.95)
        self.scaler = StandardScaler()
        
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
            if self.sentiment_analyzer:
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                sent_scores = [sentiment['compound'], sentiment['pos'], 
                              sentiment['neg'], sentiment['neu']]
            else:
                sent_scores = [0, 0, 0, 0]
            
            words = preprocess(text)
            
            try:
                import textstat
                syllables = textstat.syllable_count(words)
                FKRA = textstat.flesch_kincaid_grade(words)
                FRE = textstat.flesch_reading_ease(words)
            except:
                syllables = len(words) * 2.5
                FKRA = 10
                FRE = 60
            
            num_chars = sum(len(w) for w in words)
            num_chars_total = len(text)
            num_terms = len(text.split())
            num_words = len(words.split())
            num_unique_terms = len(set(words.split()))
            
            num_hashtags = text.count('#')
            num_mentions = text.count('@')
            num_urls = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])', text))
            num_caps = sum(1 for c in text if c.isupper())
            caps_ratio = num_caps / (len(text) + 0.001)
            
            num_exclamation = text.count('!')
            num_question = text.count('?')
            num_dots = text.count('...')
            
            has_emoji = any(ord(c) > 127 for c in text)
            repeated_chars = len(re.findall(r'(.)\1{2,}', text))
            
            feature_vector = [
                FKRA, FRE, syllables, num_chars, num_chars_total,
                num_terms, num_words, num_unique_terms,
                *sent_scores,
                num_hashtags, num_mentions, num_urls,
                caps_ratio, num_exclamation, num_question, num_dots,
                float(has_emoji), repeated_chars
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def get_all_features(self, texts):
        """Combine TF-IDF and linguistic features"""
        # Character-level TF-IDF
        char_features = self.tfidf_char.transform(texts).toarray()
        
        # Word-level TF-IDF
        word_features = self.tfidf_word.transform(texts).toarray()
        
        # Linguistic features
        ling_features = self.get_linguistic_features(texts)
        
        # Concatenate all features
        combined = np.concatenate([char_features, word_features, ling_features], axis=1)
        return combined

# ---- Main Classifier Class ----
class AdvancedHateSpeechClassifier:
    def __init__(self):
        """Initialize the advanced classifier"""
        print("Loading Advanced Hate Speech Detection Model...")
        
        # Check if files exist
        required_files = ['final_model_heavy.pkl', 'feature_extractor.pkl', 
                         'scaler.pkl', 'model_metadata.pkl']
        
        for file in required_files:
            if not os.path.exists(file):
                print(f"ERROR: {file} not found!")
                print("Please run advanced_train.py first to train the model.")
                sys.exit(1)
        
        # Load the trained model and components
        self.model = joblib.load('final_model_heavy.pkl')
        self.feature_extractor = joblib.load('feature_extractor.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.metadata = joblib.load('model_metadata.pkl')
        
        print(f"Model type: {self.metadata['best_model_type']}")
        print(f"Model accuracy: {self.metadata['accuracy']:.4f}")
        
        # Class labels
        self.class_names = {
            0: "Hate speech",
            1: "Offensive language", 
            2: "Neither"
        }
    
    def preprocess_text(self, text):
        """Preprocess single text"""
        return preprocess(str(text))
    
    def extract_features(self, texts):
        """Extract features using the trained feature extractor"""
        if isinstance(texts, str):
            texts = [texts]
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        features = self.feature_extractor.get_all_features(processed_texts)
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def predict(self, texts):
        """Predict classes for input texts"""
        features = self.extract_features(texts)
        predictions = self.model.predict(features)
        return predictions
    
    def predict_proba(self, texts):
        """Get probability scores for each class"""
        features = self.extract_features(texts)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
        else:
            predictions = self.model.predict(features)
            probabilities = np.zeros((len(predictions), 3))
            for i, pred in enumerate(predictions):
                probabilities[i, pred] = 1.0
        
        return probabilities
    
    def predict_with_confidence(self, texts):
        """Predict with confidence scores"""
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = self.predict(texts)
        probabilities = self.predict_proba(texts)
        
        results = []
        for i, text in enumerate(texts):
            pred_class = predictions[i]
            confidence = probabilities[i, pred_class]
            
            results.append({
                'text': text,
                'prediction': self.class_names[pred_class],
                'class': pred_class,
                'confidence': confidence,
                'hate_prob': probabilities[i, 0],
                'offensive_prob': probabilities[i, 1],
                'neither_prob': probabilities[i, 2]
            })
        
        return results
    
    def analyze_text(self, text):
        """Detailed analysis of a single text"""
        result = self.predict_with_confidence([text])[0]
        
        analysis = {
            'original_text': text,
            'processed_text': self.preprocess_text(text),
            'prediction': result['prediction'],
            'confidence': f"{result['confidence']:.2%}",
            'detailed_scores': {
                'hate_speech': f"{result['hate_prob']:.2%}",
                'offensive': f"{result['offensive_prob']:.2%}",
                'neither': f"{result['neither_prob']:.2%}"
            },
            'text_stats': {
                'length': len(text),
                'word_count': len(text.split()),
                'hashtags': text.count('#'),
                'mentions': text.count('@'),
                'caps_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1)
            }
        }
        
        if result['class'] == 0:
            analysis['severity'] = 'HIGH'
            analysis['action_recommended'] = 'Remove/Block'
        elif result['class'] == 1:
            analysis['severity'] = 'MEDIUM'
            analysis['action_recommended'] = 'Warning/Review'
        else:
            analysis['severity'] = 'LOW'
            analysis['action_recommended'] = 'No action needed'
        
        return analysis

# ---- Demo Function ----
def demo():
    """Demo the advanced classifier"""
    print("="*60)
    print("ADVANCED HATE SPEECH CLASSIFIER DEMO")
    print("="*60)
    
    # Initialize classifier
    classifier = AdvancedHateSpeechClassifier()
    
    # Test tweets
    test_tweets = [
        "Good morning everyone! Have a wonderful day ahead!",
        "I disagree with your political views but respect your opinion",
        "You're so stupid, nobody likes you",
        "All [group] should be removed from this country",
        "The new policy changes are concerning but we need more discussion",
        "Can't believe how dumb some people are",
        "Looking forward to the weekend plans with family",
        "These idiots don't know what they're talking about",
        "Congratulations on your achievement! Well deserved!",
        "I hate when people do that, it's so annoying",
        "You are a fucking cunt and I will kill you.",
        "This bitch is so shit i hope she kills herself",
    ]
    
    print("\n" + "-"*60)
    print("CLASSIFICATION RESULTS")
    print("-"*60)
    
    # Classify tweets
    results = classifier.predict_with_confidence(test_tweets)
    
    for result in results:
        print(f"\nText: {result['text'][:80]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Scores - Hate: {result['hate_prob']:.1%} | "
              f"Offensive: {result['offensive_prob']:.1%} | "
              f"Neither: {result['neither_prob']:.1%}")
    
    # Detailed analysis example
    print("\n" + "="*60)
    print("DETAILED ANALYSIS EXAMPLE")
    print("="*60)
    
    analysis = classifier.analyze_text("This government is failing the people badly")
    
    print(f"\nOriginal: {analysis['original_text']}")
    print(f"Processed: {analysis['processed_text']}")
    print(f"Prediction: {analysis['prediction']}")
    print(f"Confidence: {analysis['confidence']}")
    print(f"Severity: {analysis['severity']}")
    print(f"Recommended Action: {analysis['action_recommended']}")
    print("\nDetailed Scores:")
    for key, value in analysis['detailed_scores'].items():
        print(f"  - {key}: {value}")

if __name__ == "__main__":
    demo()