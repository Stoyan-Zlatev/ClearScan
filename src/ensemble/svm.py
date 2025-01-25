import itertools
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.ensemble.optimize_clustering import extract_hog_features, extract_lbp_features
from src.common.path_utils import resolve_path


#X_train = np.load(resolve_path("/data/processed/hist/20/X_train.npy"))
#X_test = np.load(resolve_path("/data/processed/hist/20/X_test.npy"))
X_train_hog = extract_hog_features(np.load(resolve_path("/data/processed/split/X_train.npy")))
X_test_hog = extract_hog_features(np.load(resolve_path("/data/processed/split/X_test.npy")))

X_train_lbp = extract_lbp_features(np.load(resolve_path("/data/processed/split/X_train.npy")))
X_test_lbp = extract_lbp_features(np.load(resolve_path("/data/processed/split/X_test.npy")))

X_train = np.concatenate((X_train_hog, X_train_lbp), axis=1)
X_test = np.concatenate((X_test_hog, X_test_lbp), axis=1)

y_train = np.load(resolve_path("/data/processed/split/y_train.npy"))
y_test = np.load(resolve_path("/data/processed/split/y_test.npy"))

print("Training SVM...")

# Create a Bagging classifier with SVM as base estimator
bagging_svm = BaggingClassifier(estimator=SVC(), n_estimators=5, random_state=42)

# Train the model
bagging_svm.fit(X_train, y_train)

# Predict on test data
y_pred = bagging_svm.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Bagging SVM Accuracy: {accuracy:.4f}")

report = classification_report(y_test, y_pred)
print(report)
