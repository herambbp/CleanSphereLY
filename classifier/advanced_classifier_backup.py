"""
Advanced Classifier with Heavy Models and Better Text Representations
Compatible with the new training pipeline
"""

import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# For compatibility with different model types
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Text processing
import re
import nltk
from sentence_transformers import SentenceTransformer

class AdvancedHateSpeechClassifier:
    def __init__(self, model_path='final_model_heavy.pkl'):
        """
        Initialize the advanced classifier
        """
        print("Loading Advanced Hate Speech Detection Model...")
        
        # Load the trained model
        self.model = joblib.load(model_path)
        
        # Load feature extractor
        self.feature_extractor = joblib.load('feature_extractor.pkl')
        
        # Load scaler
        self.scaler = joblib.load('scaler.pkl')
        
        # Load metadata
        self.metadata = joblib.load('model_metadata.pkl')
        
        print(f"Model type: {self.metadata['best_model_type']}")
        print(f"Model accuracy: {self.metadata['accuracy']:.4f}")
        
        # Class labels
        self.class_names = {
            0: "Hate speech",
            1: "Offensive language", 
            2: "Neither"
        }
        
        # Ensure NLTK resources
        self._ensure_nltk()
        
    def _ensure_nltk(self):
        """Download required NLTK resources"""
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
    
    def preprocess_text(self, text):
        """Preprocess single text"""
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Replace URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                     'URLHERE', text)
        
        # Replace mentions
        text = re.sub(r'@[\w\-]+', 'MENTIONHERE', text)
        
        # Replace multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_features(self, texts):
        """
        Extract features using the trained feature extractor
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Get all features from the feature extractor
        features = self.feature_extractor.get_all_features(
            processed_texts, 
            use_word2vec=True,
            use_bert=True
        )
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def predict(self, texts):
        """
        Predict classes for input texts
        """
        # Extract features
        features = self.extract_features(texts)
        
        # Make predictions
        predictions = self.model.predict(features)
        
        return predictions
    
    def predict_proba(self, texts):
        """
        Get probability scores for each class
        """
        # Extract features
        features = self.extract_features(texts)
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
        else:
            # For models without predict_proba, use decision function
            predictions = self.model.predict(features)
            # Convert to one-hot style probabilities
            probabilities = np.zeros((len(predictions), 3))
            for i, pred in enumerate(predictions):
                probabilities[i, pred] = 1.0
        
        return probabilities
    
    def predict_with_confidence(self, texts):
        """
        Predict with confidence scores
        """
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
    
    def batch_classify(self, texts, batch_size=32):
        """
        Classify texts in batches for efficiency
        """
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            results = self.predict_with_confidence(batch)
            all_results.extend(results)
        
        return all_results
    
    def analyze_text(self, text):
        """
        Detailed analysis of a single text
        """
        result = self.predict_with_confidence([text])[0]
        
        # Add detailed analysis
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
                'urls': len(re.findall(r'http[s]?://\S+', text)),
                'caps_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1)
            }
        }
        
        # Determine severity
        if result['class'] == 0:  # Hate speech
            analysis['severity'] = 'HIGH'
            analysis['action_recommended'] = 'Remove/Block'
        elif result['class'] == 1:  # Offensive
            analysis['severity'] = 'MEDIUM'
            analysis['action_recommended'] = 'Warning/Review'
        else:  # Neither
            analysis['severity'] = 'LOW'
            analysis['action_recommended'] = 'No action needed'
        
        return analysis

# Backward compatibility functions
def get_tweets_predictions(tweets, perform_prints=True):
    """
    Backward compatible function for existing code
    """
    if perform_prints:
        print(f"{len(tweets)} tweets to classify")
    
    classifier = AdvancedHateSpeechClassifier()
    predictions = classifier.predict(tweets)
    
    return predictions

def class_to_name(class_label):
    """
    Convert class label to name
    """
    return {
        0: "Hate speech",
        1: "Offensive language",
        2: "Neither"
    }.get(class_label, "Unknown")

# Demo and testing
def demo():
    """
    Demo the advanced classifier
    """
    print("="*60)
    print("ADVANCED HATE SPEECH CLASSIFIER DEMO")
    print("="*60)
    
    # Initialize classifier
    classifier = AdvancedHateSpeechClassifier()
    
    # Test tweets
    test_tweets = [
        "Good morning everyone! Have a wonderful day ahead! 😊",
        "I disagree with your political views but respect your opinion",
        "You're so stupid, nobody likes you",
        "All [group] should be removed from this country",
        "The new policy changes are concerning but we need more discussion",
        "Can't believe how dumb some people are",
        "Looking forward to the weekend plans with family",
        "These idiots don't know what they're talking about",
        "Congratulations on your achievement! Well deserved!",
        "I hate when people do that, it's so annoying"
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
    
    # Detailed analysis of one tweet
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
    print("\nText Statistics:")
    for key, value in analysis['text_stats'].items():
        print(f"  - {key}: {value}")

if __name__ == "__main__":
    # Check if model exists
    import os
    if os.path.exists('final_model_heavy.pkl'):
        demo()
    else:
        print("Model not found. Please run advanced_train.py first to train the model.")
        print("\nFor backward compatibility testing with old model:")
        
        # Try old model if it exists
        if os.path.exists('final_model.pkl'):
            print("Testing with existing logistic regression model...")
            
            # Test backward compatibility
            import pandas as pd
            try:
                df = pd.read_csv("trump_tweets.csv", encoding="latin-1", on_bad_lines="skip")
                tweets = df['Text'].dropna().head(5).tolist()
                
                predictions = get_tweets_predictions(tweets)
                for tweet, pred in zip(tweets, predictions):
                    print(f"\n{tweet[:100]}...")
                    print(f"=> {class_to_name(pred)}")
            except Exception as e:
                print(f"Error testing: {e}")