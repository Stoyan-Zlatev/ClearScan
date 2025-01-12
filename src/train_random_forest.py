# train_random_forest.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def train_random_forest_classifier(histograms, labels):
    """
    Train a Random Forest Classifier using histograms as input features.

    Args:
        histograms: Histogram features extracted from the images.
        labels: Corresponding labels for the histograms.

    Returns:
        model: Trained Random Forest model.
    """
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(histograms, labels, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    return model