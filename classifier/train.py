import numpy as np
import pandas as pd
import joblib
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re

# ---- NLTK resources (handle both old/new names)
def _ensure_nltk():
    import nltk
    def _dl(name): 
        try: nltk.download(name, quiet=True)
        except: pass

    _dl("stopwords"); _dl("punkt")
    # try new tagger first, then legacy
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        _dl("averaged_perceptron_tagger_eng")
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        _dl("averaged_perceptron_tagger")

_ensure_nltk()

import nltk
STOPWORDS = nltk.corpus.stopwords.words('english')
STOPWORDS.extend(["#ff", "ff", "rt"])


STOPWORDS = nltk.corpus.stopwords.words('english')
STOPWORDS.extend(["#ff", "ff", "rt"])
stemmer = PorterStemmer()

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
    return [stemmer.stem(t) for t in tweet.split()]

def basic_tokenize(tweet: str):
    tweet = " ".join(re.split(r"[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

def get_pos_tags(texts):
    out = []
    for t in texts:
        tokens = basic_tokenize(preprocess(t))
        try:
            tags = nltk.pos_tag(tokens, lang="eng")  # new models prefer lang
        except TypeError:
            tags = nltk.pos_tag(tokens)  # fallback for older NLTK
        out.append(" ".join([p for (_, p) in tags]))
    return out


# ---- Load data
df = pd.read_csv("../data/labeled_data.csv", encoding="latin-1", on_bad_lines="skip")
df = df.dropna(subset=["tweet", "class"])
X_text = df["tweet"].astype(str).tolist()
y = df["class"].astype(int).values

# ---- Vectorizers
tf_vectorizer = TfidfVectorizer(
    tokenizer=tokenize_stem,
    stop_words=STOPWORDS,
    ngram_range=(1,3),
    min_df=5,
    max_df=0.95,
    use_idf=False,
    norm=None,
)

idf_vectorizer = TfidfVectorizer(
    tokenizer=tokenize_stem,
    stop_words=STOPWORDS,
    ngram_range=(1,3),
    min_df=5,
    max_df=0.95,
    use_idf=True,
    norm=None,
)

pos_vectorizer = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"(?u)\b\w+\b",
    ngram_range=(1,3),
    min_df=2,
    use_idf=False,
    norm=None,
)

print("Fitting TF vectorizer...")
tf_matrix = tf_vectorizer.fit_transform(X_text)

print("Fitting IDF vectorizer (aligned vocab)...")
idf_vectorizer.fit(X_text)

# align IDF vector to tf_vectorizer vocabulary
vocab = tf_vectorizer.vocabulary_
idf_full = idf_vectorizer.idf_
idf_vocab = idf_vectorizer.vocabulary_
idf_vec = np.ones(len(vocab), dtype=float)
for token, idx in vocab.items():
    j = idf_vocab.get(token)
    if j is not None:
        idf_vec[idx] = idf_full[j]

print("Fitting POS vectorizer...")
pos_tags = get_pos_tags(X_text)
pos_matrix = pos_vectorizer.fit_transform(pos_tags)

# ---- Other features
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
import textstat
vs = VS()

def other_features_(tweet: str):
    sentiment = vs.polarity_scores(tweet)
    words = preprocess(tweet)
    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = (syllables + 0.001) / (num_words + 0.001)
    num_unique_terms = len(set(words.split()))
    FKRA = 0.39*(num_words/1.0) + 11.8*avg_syl - 15.59
    FRE  = 206.835 - 1.015*(num_words/1.0) - (84.6*avg_syl)
    # counts
    space_pattern = r'\s+'
    giant_url_regex = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       r'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = r'@[\w\-]+'
    hashtag_regex = r'#[\w\-]+'
    parsed = re.sub(space_pattern, ' ', tweet)
    parsed = re.sub(giant_url_regex, 'URLHERE', parsed)
    parsed = re.sub(mention_regex, 'MENTIONHERE', parsed)
    parsed = re.sub(hashtag_regex, 'HASHTAGHERE', parsed)
    url_count = parsed.count('URLHERE')
    mention_count = parsed.count('MENTIONHERE')
    hashtag_count = parsed.count('HASHTAGHERE')
    return [FKRA, FRE, syllables, num_chars, num_chars_total, num_terms,
            num_words, num_unique_terms, sentiment['compound'],
            hashtag_count, mention_count]

print("Computing other features...")
oth = np.vstack([other_features_(t) for t in X_text])

# ---- Final matrix: (tf * idf_vec) + pos + other
tfidf_matrix = tf_matrix.toarray() * idf_vec
M = np.concatenate([tfidf_matrix, pos_matrix.toarray(), oth], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    M, y, test_size=0.2, random_state=42, stratify=y
)

print("Training LogisticRegression...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(classification_report(y_test, pred))
print("Accuracy:", accuracy_score(y_test, pred))

print("Saving artifacts...")
joblib.dump(clf, "final_model.pkl")
joblib.dump(tf_vectorizer, "final_tfidf.pkl")
joblib.dump(idf_vec, "final_idf.pkl")
joblib.dump(pos_vectorizer, "final_pos.pkl")
print("Done.")
