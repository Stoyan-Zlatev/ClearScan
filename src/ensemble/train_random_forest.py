import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold

def train_random_forest_classifier(train_histograms, train_labels):
    print("Creating classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train_histograms, train_labels)
    return model

def evaluate_model(model, test_histograms, test_labels, results_file_path=None):
    print("Evaluating model...")
    y_pred = model.predict(test_histograms)

    report = "Classification Report:\n"
    report = report + str(classification_report(test_labels, y_pred))
    report = report + f"\nAccuracy: {accuracy_score(test_labels, y_pred):.2f}\n"

    print(report)
    if results_file_path is not None:
        with open(results_file_path, "w") as f:
            f.write(report)


def perform_k_fold_cross_validation(histograms, labels, n_splits=10, results_file_path=None):
    print(f"Performing {n_splits}-fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    accuracies = []
    report = ""

    for fold, (train_idx, test_idx) in enumerate(skf.split(histograms, labels), 1):
        X_train, X_test = histograms[train_idx], histograms[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        report = report + f"Fold {fold} Accuracy: {accuracy:.2f}\n"

    report = report + "\nCross-Validation Results:\n"
    report = report + f"\nMean Accuracy: {np.mean(accuracies):.2f}"
    report = report + f"\nStandard Deviation: {np.std(accuracies):.2f}"

    print(report)
    if results_file_path is not None:
        with open(results_file_path, "w") as f:
            f.write(report)