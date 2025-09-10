from advanced_classifier import AdvancedHateSpeechClassifier

# Initialize once
classifier = AdvancedHateSpeechClassifier()

# Classify single tweet
result = classifier.analyze_text("You fucking cunt i will kill you ")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")

# Classify multiple tweets
# tweets = ["tweet1", "tweet2", "tweet3"]
# results = classifier.predict_with_confidence(tweets)
# for r in results:
#     print(f"{r['text']}: {r['prediction']} ({r['confidence']:.2%})")