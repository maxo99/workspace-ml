import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import seaborn as sns

class SpamClassifier:
    is_trained: bool

    def __init__(
        self,
        max_features=1000,
        random_state=42,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            lowercase=True,
        )
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver="lbfgs",
        )
        self.is_trained = False

    def prepare_data(self, texts, labels):
        features = self.vectorizer.fit_transform(texts)
        return features, np.array(labels)

    def train(self, X_train, y_train):
        print("Training model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training complete!")

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Train the model first!")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Train the model first!")
        return self.model.predict_proba(X)

    def predict_text(self, texts):
        features = self.vectorizer.transform(texts)
        predictions = self.predict(features)
        probabilities = self.predict_proba(features)

        results = []
        for i, text in enumerate(texts):
            results.append(
                {
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "prediction": "SPAM" if predictions[i] == 1 else "HAM",
                    "spam_probability": probabilities[i][1],
                    "confidence": max(probabilities[i]),
                }
            )
        return results


class ModelEvaluator:
    @staticmethod
    def evaluate_model(y_true, y_pred, y_proba=None):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
        }

        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])

        return metrics

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"],
        )
        plt.title("Confusion Matrix - Spam Classification", fontsize=14, pad=20)
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Confusion matrix saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_roc_curve(y_true, y_proba, save_path="roc_curve.png"):
        fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
        auc = roc_auc_score(y_true, y_proba[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve - Spam Classification", fontsize=14, pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📈 ROC curve saved to {save_path}")
        plt.close()

    @staticmethod
    def print_classification_report(y_true, y_pred):
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"], digits=4))
