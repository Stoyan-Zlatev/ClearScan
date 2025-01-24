import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.common.path_utils import resolve_path


#X_train = np.load(resolve_path("/data/processed/hist/20/X_train.npy"))
#X_test = np.load(resolve_path("/data/processed/hist/20/X_test.npy"))
X_train = np.load(resolve_path("/data/processed/split/X_train.npy"))
X_test = np.load(resolve_path("/data/processed/split/X_test.npy"))

y_train = np.load(resolve_path("/data/processed/split/y_train.npy"))
y_test = np.load(resolve_path("/data/processed/split/y_test.npy"))

X_train_avg = []
for i in range(X_train.shape[0]):
    X_train_avg.append(np.mean(X_train[i], axis=0))

X_test_avg = []
for i in range(X_test.shape[0]):
    X_test_avg.append(np.mean(X_test[i], axis=0))

# Create a Bagging classifier with SVM as base estimator
bagging_svm = BaggingClassifier(estimator=SVC(), n_estimators=50, random_state=42)

# Train the model
bagging_svm.fit(X_train_avg, y_train)

# Predict on test data
y_pred = bagging_svm.predict(X_test_avg)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Bagging SVM Accuracy: {accuracy:.4f}")

report = classification_report(y_test, y_pred)
print(report)
