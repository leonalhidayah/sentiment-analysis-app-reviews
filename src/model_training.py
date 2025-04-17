import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def plot_confusion_matrix(cm):
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
    )
    plt.title("Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.show()


# Function to train a single model
def train_model(model, X_train, y_train):
    """
    Train a given model and return the training time.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time


# Function to evaluate a model
def evaluate_model(model, X, y):
    """
    Evaluate a classification model using accuracy, precision, recall, F1, and confusion matrix.
    """
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred)
    return accuracy, precision, recall, f1, cm, cr


# Function to print evaluation results
def print_evaluation_results(model_name, train_metrics, test_metrics):
    """
    Print the evaluation results for both training and testing datasets.
    """
    print("=" * 100)
    print(f" {model_name} Classifier ".center(100, "="))
    print("=" * 100)

    print("Training")
    print(f"Train Accuracy: {train_metrics['Accuracy']}")
    print(f"Train Precision: {train_metrics['Precision']}")
    print(f"Train Recall: {train_metrics['Recall']}")
    print(f"Train F1-Score: {train_metrics['F1']}")
    print("Confusion Matrix:")
    plot_confusion_matrix(train_metrics["Confusion Matrix"])

    print("\nTesting")
    print(f"Test Accuracy: {test_metrics['Accuracy']}")
    print(f"Test Precision: {test_metrics['Precision']}")
    print(f"Test Recall: {test_metrics['Recall']}")
    print(f"Test F1-Score: {test_metrics['F1']}")
    print("Classification Report: ")
    print(test_metrics["Classification Report"])
    print("Confusion Matrix:")
    plot_confusion_matrix(test_metrics["Confusion Matrix"])


# Main function to train and evaluate multiple models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, models):
    """
    Train and evaluate multiple classification models and return a summary of evaluation results.
    """
    evaluation_results = []

    for model_name, model in models.items():
        # Train the model
        trained_model, training_time = train_model(model, X_train, y_train)

        # Evaluate on training data
        train_accuracy, train_precision, train_recall, train_f1, train_cm, train_cr = (
            evaluate_model(trained_model, X_train, y_train)
        )

        # Evaluate on testing data
        test_accuracy, test_precision, test_recall, test_f1, test_cm, test_cr = (
            evaluate_model(trained_model, X_test, y_test)
        )

        # Print results
        print_evaluation_results(
            model_name,
            {
                "Accuracy": train_accuracy,
                "Precision": train_precision,
                "Recall": train_recall,
                "F1": train_f1,
                "Confusion Matrix": train_cm,
                "Classification Report": train_cr,
            },
            {
                "Accuracy": test_accuracy,
                "Precision": test_precision,
                "Recall": test_recall,
                "F1": test_f1,
                "Confusion Matrix": test_cm,
                "Classification Report": test_cr,
            },
        )

        # Store results
        evaluation_results.append(
            {
                "Model": model_name,
                "Train Accuracy": train_accuracy,
                "Test Accuracy": test_accuracy,
                "Train Precision": train_precision,
                "Test Precision": test_precision,
                "Train Recall": train_recall,
                "Test Recall": test_recall,
                "Train F1": train_f1,
                "Test F1": test_f1,
                "Training Time (seconds)": training_time,
            }
        )

    result_df = pd.DataFrame(evaluation_results)
    return result_df


def save_model(model, file_path):
    """
    Save the trained model to a file.
    """
    import joblib

    joblib.dump(model, file_path)
