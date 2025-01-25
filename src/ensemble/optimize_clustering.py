# Re-importing necessary libraries and re-executing due to a reset environment
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
import numpy as np


from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from sklearn.cluster import KMeans
import multiprocessing
from skimage.feature import hog, local_binary_pattern

from src.common.path_utils import resolve_path

# Parallel SIFT Feature Extraction
def extract_sift_features(images, num_threads=None):

    def process_image(img):
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return descriptors

    sift = cv2.SIFT_create()

    if num_threads is None:
        num_threads = multiprocessing.cpu_count() // 2

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_image, images))

    # Filter out None results and stack descriptors
    all_descriptors = [desc for desc in results if desc is not None]
    return all_descriptors

# Add HOG and LBP features
def extract_hog_features(images, num_threads=None):
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    """
    def process_image(img):
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        return features

    if num_threads is None:
        num_threads = multiprocessing.cpu_count() // 2

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_image, images))

    all_descriptors = [desc for desc in results if desc is not None]
    return all_descriptors


def extract_lbp_features(images, num_threads=None):
    """
    Extract Local Binary Pattern (LBP) features.
    """
    def process_image(img):
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        features = local_binary_pattern(gray, P=8, R=1.0, method="uniform")
        hist, _ = np.histogram(features.ravel(), bins=np.arange(0, 10), range=(0, 9))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6) # Normalize histogram
        return hist

    if num_threads is None:
        num_threads = multiprocessing.cpu_count() // 2

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_image, images))

    all_descriptors = [desc for desc in results if desc is not None]
    return all_descriptors

# Parallel K-Means Clustering
def create_visual_vocabulary(descriptors, k=20):
    """
    Perform k-Means clustering in parallel.
    """
    all_descriptors = np.vstack(descriptors)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(all_descriptors)
    return kmeans

# Unified Feature Extraction Pipeline
def extract_combined_features(images):
    """
    Extract combined SIFT, HOG, and LBP features for richer representation.
    """
    sift_features = extract_sift_features(images)
    hog_features = extract_hog_features(images)
    lbp_features = extract_lbp_features(images)

    combined_features = []
    for sift, hog, lbp in zip(sift_features, hog_features, lbp_features):
        combined = np.hstack([
            np.mean(sift, axis=0) if sift is not None else np.zeros(128), # Mean SIFT descriptor
            hog,
            lbp
        ])
        combined_features.append(combined)

    return np.array(combined_features)

def save_features():
    X_train_raw = np.load(resolve_path("/data/processed/split/X_train.npy"))
    X_test_raw = np.load(resolve_path("/data/processed/split/X_test.npy"))
    #X_val_raw = np.load(resolve_path("/data/processed/split/X_verify.npy"))

    print("Extracting features...")
    X_train_descriptors = extract_lbp_features(X_train_raw)
    X_test_descriptors = extract_lbp_features(X_test_raw)
    #X_val_descriptors = extract_sift_features(X_val_raw)

    os.makedirs(resolve_path("/data/processed/feat"), exist_ok=True)
    np.save(resolve_path("/data/processed/feat/X_train.npy"), X_train_descriptors)
    np.save(resolve_path("/data/processed/feat/X_test.npy"), X_test_descriptors)
    #np.save(resolve_path("/data/processed/feat/X_verify.npy"), X_val_descriptors)


print("Extracting features...")
save_features()
#exit()
print("Loading data...")
y_train = np.load(resolve_path("/data/processed/split/y_train.npy"))
y_test = np.load(resolve_path("/data/processed/split/y_test.npy"))
#y_val = np.load(resolve_path("/data/processed/split/y_verify.npy"))

X_train_descriptors = np.load(resolve_path("/data/processed/feat/X_train.npy"))
X_test_descriptors = np.load(resolve_path("/data/processed/feat/X_test.npy"))
#X_val_descriptors = np.load(resolve_path("/data/processed/feat/X_verify.npy"))

# Define a parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

print("Starting grid search...")

# Grid search with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_descriptors, y_train)

# Best hyperparameters
best_params = grid_search.best_params_

print(str(best_params))

print("Evaluating best model...")

# Evaluate on test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test_descriptors)
classification_rep = classification_report(y_test, y_pred)

print("Results...")
print(str(best_params))
print(str(classification_rep))

