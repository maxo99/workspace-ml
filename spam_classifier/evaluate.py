from spam_classifier.data_builder import create_sample_dataset, split_data
from spam_classifier.helpers import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from spam_classifier.model import ModelEvaluator, SpamClassifier

# Create dataset
df = create_sample_dataset()

# Split data
X_train_text, X_test_text, y_train, y_test = split_data(df)
print(f"Train set: {len(X_train_text)} emails")
print(f"Test set: {len(X_test_text)} emails")



# Initialize and prepare Classifier
classifier = SpamClassifier(max_features=1000)
X_train, _ = classifier.prepare_data(X_train_text, y_train)
X_test = classifier.vectorizer.transform(X_test_text)
print("Feature extraction complete")
print(f"   - Feature dimensions: {X_train.shape[1]}")


# Train the Classifier
classifier.train(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)

# Evaluate Model
print("\n📊 Step 6: Evaluating model performance...")
evaluator = ModelEvaluator()
roc_auc = calculate_metrics(y_test, y_pred, y_proba)
metrics = evaluator.evaluate_model(y_test, y_pred, y_proba)
print("\nPerformance Metrics:")
print(f"  • Accuracy:  {metrics['accuracy']:.4f}")
print(f"  • Precision: {metrics['precision']:.4f}")
print(f"  • Recall:    {metrics['recall']:.4f}")
print(f"  • F1-Score:  {metrics['f1_score']:.4f}")
print(f"  • ROC-AUC:   {metrics['roc_auc']:.4f}")

# Step 7: Test with new emails
print("\n🧪 Step 7: Testing with new email samples...")
new_emails = [
    "Hi, let's meet for coffee tomorrow to discuss the project.",
    "WINNER! You've been selected for a FREE vacation package!",
    "Please review the attached document at your earliest convenience.",
    "URGENT! Your bank account has been compromised! Click here NOW!",
]

results = classifier.predict_text(new_emails)

print("\nPrediction Results:")
print("-" * 70)
for i, result in enumerate(results, 1):
    print(f"\nEmail {i}: {result['text']}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Spam Probability: {result['spam_probability']:.4f}")
    print(f"  Confidence: {result['confidence']:.4f}")

print("\n" + "="*70)
print("✅ SPAM CLASSIFIER COMPLETE!")
print("="*70)
print("\n📚 Next Steps:")
print("  1. Experiment with different thresholds for classification")
print("  2. Try adding more sophisticated features (email headers, links, etc.)")
print("  3. Collect real email data to improve the model")
print("  4. Move on to Day 50: Multi-Class Classification")
print("\n🎯 You've built a production-ready binary classifier!")
