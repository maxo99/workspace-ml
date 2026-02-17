import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# Calculate metrics
def calculate_metrics(
    _y_test,
    _y_pred,
    _y_proba,
):
    accuracy = accuracy_score(_y_test, _y_pred)
    precision = precision_score(_y_test, _y_pred)
    recall = recall_score(_y_test, _y_pred)
    f1 = f1_score(_y_test, _y_pred)
    roc_auc = roc_auc_score(_y_test, _y_proba[:, 1])

    print("\nPerformance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    return roc_auc


# Generate confusion matrix
def plot_confusion_matrix(
    _y_test,
    _y_pred,
):
    cm = confusion_matrix(_y_test, _y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
    )
    plt.title("Confusion Matrix - Spam Classification")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")


# Generate ROC curve
def plot_roc_curve(y_test, y_proba, roc_auc):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Spam Classification")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")


