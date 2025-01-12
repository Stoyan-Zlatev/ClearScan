# test_random_forest.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score


def test_random_forest_classifier(histograms, labels, n_splits=10):
    """
    Test a Random Forest Classifier using 10-fold stratified cross-validation.

    Args:
        histograms: Histogram features extracted from the images.
        labels: Corresponding labels for the histograms.
        n_splits: Number of folds for stratified cross-validation.

    Returns:
        None: Prints validation results.
    """
    print(f"Performing {n_splits}-fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(histograms, labels), 1):
        X_train, X_test = histograms[train_idx], histograms[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        print(f"Fold {fold} Accuracy: {accuracy:.2f}")

    print("\nCross-Validation Results:")
    print(f"Mean Accuracy: {np.mean(accuracies):.2f}")
    print(f"Standard Deviation: {np.std(accuracies):.2f}")

    # Optional: Train final model on all data and display metrics
    print("\nTraining final model on entire dataset...")
    model.fit(histograms, labels)
    y_pred_final = model.predict(histograms)
    print("Final Model Classification Report:")
    print(classification_report(labels, y_pred_final))