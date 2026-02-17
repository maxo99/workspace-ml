from spam_classifier.data_builder import create_sample_dataset, split_data
from spam_classifier.model import SpamClassifier

# Create dataset
df = create_sample_dataset()

# Split data
X_train_text, X_test_text, y_train, y_test = split_data(df)


# Initialize and train
classifier = SpamClassifier(max_features=1000)
X_train, _ = classifier.prepare_data(X_train_text, y_train)
X_test = classifier.vectorizer.transform(X_test_text)


classifier.train(X_train, y_train)

