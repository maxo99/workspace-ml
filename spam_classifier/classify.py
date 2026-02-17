from spam_classifier.model import SpamClassifier


new_emails = [
    "Hi, let's meet for coffee tomorrow to discuss the project.",
    "WINNER! You've been selected for a FREE vacation package!",
    "Please review the attached document at your earliest convenience.",
    "URGENT! Your bank account has been compromised! Click here NOW!",
]

classifier = SpamClassifier(max_features=1000)
results = classifier.predict_text(new_emails)

print("\nReal-Time Predictions:")
for i, result in enumerate(results, 1):
    print(f"\nEmail {i}: {result['text']}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Spam Probability: {result['spam_probability']:.4f}")
    print(f"  Confidence: {result['confidence']:.4f}")


# Expected output:
# ```
# Real-Time Predictions:

# Email 1: Hi, let's meet for coffee tomorrow to discuss...
#   Prediction: HAM
#   Spam Probability: 0.1234
#   Confidence: 0.8766

# Email 2: WINNER! You've been selected for a FREE vacat...
#   Prediction: SPAM
#   Spam Probability: 0.9523
#   Confidence: 0.9523