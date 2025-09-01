"""
Classifier runner (Python 3 compatible)

- Loads pre-trained artifacts:
    final_model.pkl, final_tfidf.pkl, final_idf.pkl, final_pos.pkl
- Transforms input texts into model features
- Predicts class labels: {0: Hate speech, 1: Offensive language, 2: Neither}

This file is resilient to:
- Old scikit-learn/joblib pickle module paths
- Vectorizers saved with custom tokenizer from __main__
- Missing/incompatible POS vectorizer (falls back to 0-column dummy)
"""

from __future__ import annotations

import sys
import types
import re
from typing import List, Sequence

import numpy as np
import pandas as pd

# ---- Back-compat shims for legacy pickles -----------------------------------
# Some old pickles reference sklearn.externals.joblib and other old paths.
import joblib as _joblib
mod_joblib = types.ModuleType("sklearn.externals.joblib")
mod_joblib.__dict__.update(_joblib.__dict__)
sys.modules["sklearn.externals.joblib"] = mod_joblib

try:
    import sklearn.svm as _svm
    mod_svm_classes = types.ModuleType("sklearn.svm.classes")
    mod_svm_classes.LinearSVC = _svm.LinearSVC
    sys.modules["sklearn.svm.classes"] = mod_svm_classes
except Exception:
    pass

try:
    import sklearn.linear_model as _lin
    mod_logistic = types.ModuleType("sklearn.linear_model.logistic")
    mod_logistic.LogisticRegression = _lin.LogisticRegression
    sys.modules["sklearn.linear_model.logistic"] = mod_logistic
except Exception:
    pass
# -----------------------------------------------------------------------------

# Normal imports (kept for compatibility)
from sklearn.svm import LinearSVC  # noqa: F401
from sklearn.linear_model import LogisticRegression  # noqa: F401
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
from sklearn.feature_selection import SelectFromModel  # noqa: F401

import nltk
from nltk.stem.porter import PorterStemmer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
import textstat


# -----------------------------
# Globals / resources
# -----------------------------
try:
    STOPWORDS = nltk.corpus.stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = nltk.corpus.stopwords.words("english")
STOPWORDS.extend(["#ff", "ff", "rt"])

STEM = PorterStemmer()
SENT = VS()


# -----------------------------
# Text processing helpers
# -----------------------------
def preprocess(text_string: str) -> str:
    """
    Replace:
      - URLs -> URLHERE
      - multiple whitespace -> single space
      - mentions -> MENTIONHERE
    """
    space_pattern = r"\s+"
    giant_url_regex = (
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
        r"[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    mention_regex = r"@[\w\-]+"

    parsed_text = re.sub(space_pattern, " ", str(text_string))
    parsed_text = re.sub(giant_url_regex, "URLHERE", parsed_text)
    parsed_text = re.sub(mention_regex, "MENTIONHERE", parsed_text)
    return parsed_text


def tokenize_stem(tweet: str) -> List[str]:
    """Lowercase, strip non-letters, remove stopwords, stem tokens."""
    tweet = " ".join(re.split(r"[^a-zA-Z]*", str(tweet).lower())).strip()
    toks = [t for t in tweet.split() if t and t not in STOPWORDS]
    return [STEM.stem(t) for t in toks]


def basic_tokenize(tweet: str) -> List[str]:
    """Lowercase, keep basic punctuation, no stemming."""
    tweet = " ".join(re.split(r"[^a-zA-Z.,!?]*", str(tweet).lower())).strip()
    return [t for t in tweet.split() if t]


# Make custom tokenizer discoverable for pickles saved with __main__.tokenize_stem
try:
    setattr(sys.modules.get("__main__", sys.modules[__name__]), "tokenize_stem", tokenize_stem)
except Exception:
    pass


def get_pos_tags(tweets: Sequence[str]) -> List[str]:
    """Return POS tag strings per tweet. (Slow; only used if pos_vectorizer is real)"""
    # Ensure taggers exist (both new and old names)
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger", quiet=True)
        except Exception:
            pass

    tweet_tags: List[str] = []
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        try:
            tags = nltk.pos_tag(tokens, lang="eng")
        except TypeError:
            tags = nltk.pos_tag(tokens)
        tag_list = [pos for (_, pos) in tags]
        tweet_tags.append(" ".join(tag_list))
    return tweet_tags


def count_twitter_objs(text_string: str) -> tuple[int, int, int]:
    """
    Return counts: (#urls, #mentions, #hashtags)
    """
    space_pattern = r"\s+"
    giant_url_regex = (
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
        r"[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    mention_regex = r"@[\w\-]+"
    hashtag_regex = r"#[\w\-]+"

    parsed_text = re.sub(space_pattern, " ", str(text_string))
    parsed_text = re.sub(giant_url_regex, "URLHERE", parsed_text)
    parsed_text = re.sub(mention_regex, "MENTIONHERE", parsed_text)
    parsed_text = re.sub(hashtag_regex, "HASHTAGHERE", parsed_text)

    return (
        parsed_text.count("URLHERE"),
        parsed_text.count("MENTIONHERE"),
        parsed_text.count("HASHTAGHERE"),
    )


def other_features_(tweet: str) -> List[float]:
    """
    Sentiment, readability, and simple count features.
    """
    sentiment = SENT.polarity_scores(tweet)
    words = preprocess(tweet)

    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round((syllables + 0.001) / (num_words + 0.001), 4)
    num_unique_terms = len(set(words.split()))

    # Modified FK grade & FRE (assume one sentence)
    FKRA = round(0.39 * (num_words / 1.0) + 11.8 * avg_syl - 15.59, 1)
    FRE = round(206.835 - 1.015 * (num_words / 1.0) - (84.6 * avg_syl), 2)

    url_count, mention_count, hashtag_count = count_twitter_objs(tweet)

    return [
        FKRA,
        FRE,
        float(syllables),
        float(num_chars),
        float(num_chars_total),
        float(num_terms),
        float(num_words),
        float(num_unique_terms),
        float(sentiment["compound"]),
        float(hashtag_count),
        float(mention_count),
    ]


def get_oth_features(tweets: Sequence[str]) -> np.ndarray:
    return np.array([other_features_(t) for t in tweets])


def _ensure_str_list(seq: Sequence[object]) -> List[str]:
    fixed: List[str] = []
    for s in seq:
        if isinstance(s, bytes):
            fixed.append(s.decode("utf-8", "ignore"))
        elif isinstance(s, str):
            fixed.append(s)
        else:
            fixed.append(str(s))
    return fixed


# -----------------------------
# Model I/O + transforms
# -----------------------------
def transform_inputs(
    tweets: Sequence[str],
    tf_vectorizer,
    idf_vector: np.ndarray,
    pos_vectorizer,
) -> pd.DataFrame:
    """
    Use *pre-trained* vectorizers to produce features:
     - TF-IDF n-grams (tf * idf_vector)
     - POS tag n-grams (or dummy zeros)
     - Other numeric features
    """
    tweets = _ensure_str_list(tweets)

    # Text features
    tf_array = tf_vectorizer.transform(tweets).toarray()
    tfidf_array = tf_array * idf_vector
    print("Built TF-IDF array")

    # POS features: support dummy
    if hasattr(pos_vectorizer, "is_dummy") and getattr(pos_vectorizer, "is_dummy"):
        pos_array = np.zeros((len(tweets), 0), dtype=float)
        print("POS disabled (dummy vectorizer)")
    else:
        pos_tags = get_pos_tags(tweets)
        pos_array = pos_vectorizer.transform(pos_tags).toarray()
        print("Built POS array")

    # Other features
    oth_array = get_oth_features(tweets)
    print("Built other feature array")

    M = np.concatenate([tfidf_array, pos_array, oth_array], axis=1)
    return pd.DataFrame(M)


def predictions(X: pd.DataFrame, model) -> np.ndarray:
    return model.predict(X)


def class_to_name(class_label: int) -> str:
    return {
        0: "Hate speech",
        1: "Offensive language",
        2: "Neither",
    }.get(class_label, "No label")


def get_tweets_predictions(tweets: Sequence[object], perform_prints: bool = True) -> np.ndarray:
    tweets = _ensure_str_list(tweets)
    if perform_prints:
        print(len(tweets), "tweets to classify")

    if perform_prints:
        print("Loading trained classifier...")
    model = _joblib.load("final_model.pkl")

    if perform_prints:
        print("Loading other information...")
    tf_vectorizer = _joblib.load("final_tfidf.pkl")
    idf_vector = _joblib.load("final_idf.pkl")

    # Try to load POS vectorizer; if fails, use dummy (0 columns)
    try:
        pos_vectorizer = _joblib.load("final_pos.pkl")
    except Exception:
        class _DummyPOS:
            is_dummy = True
            def transform(self, X):
                return np.zeros((len(X), 0), dtype=float)
        pos_vectorizer = _DummyPOS()
        print("POS vectorizer missing/incompatible — using dummy (0 columns).")

    if perform_prints:
        print("Transforming inputs...")
    X = transform_inputs(tweets, tf_vectorizer, idf_vector, pos_vectorizer)

    if perform_prints:
        print("Running classification model...")
    return predictions(X, model)


# -----------------------------
# CLI entry (optional demo)
# -----------------------------
if __name__ == "__main__":
    print("Loading data to classify...")

    # Demo classification if trump_tweets.csv exists
    try:
        df_demo = pd.read_csv("trump_tweets.csv", encoding="latin-1", on_bad_lines="skip")
        trump_tweets = [x for x in df_demo.Text if isinstance(x, str)]
        if trump_tweets:
            preds = get_tweets_predictions(trump_tweets)
            print("Printing predicted values: ")
            for i, t in enumerate(trump_tweets[:20]):  # cap output
                print(t)
                print(class_to_name(int(preds[i])))
    except Exception as e:
        print("Skipping demo classification:", e)

    # Accuracy on labeled data (optional)
    try:
        print("Calculate accuracy on labeled data (if available)")
        df = pd.read_csv("../data/labeled_data.csv", encoding="latin-1", on_bad_lines="skip")
        tweets = [x for x in df["tweet"].values if isinstance(x, str)]
        tweets_class = df["class"].values
        preds = get_tweets_predictions(tweets, perform_prints=False)
        acc = (preds == tweets_class[: len(preds)]).sum() / float(len(tweets_class))
        print(("accuracy", acc))
    except Exception as e:
        print("Skipping accuracy calc:", e)
