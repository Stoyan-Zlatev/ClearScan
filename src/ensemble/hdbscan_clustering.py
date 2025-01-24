import os
import hdbscan
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

from src.ensemble.feature_extraction import extract_sift_features
from src.ensemble.train_random_forest import evaluate_model, perform_k_fold_cross_validation, train_random_forest_classifier
from src.common.path_utils import resolve_path


min_cluster_size = 15
min_samples = 10

#print("Loading data...")
#y_train = np.load(resolve_path("/data/processed/split/y_train.npy"))
#y_test = np.load(resolve_path("/data/processed/split/y_test.npy"))
#y_val = np.load(resolve_path("/data/processed/split/y_verify.npy"))

#X_train_descriptors = np.load(resolve_path("/data/processed/feat/X_train.npy"))
#X_test_descriptors = np.load(resolve_path("/data/processed/feat/X_test.npy"))
#X_val_descriptors = np.load(resolve_path("/data/processed/feat/X_verify.npy"))


print("Extracting features...")

X_train_raw = np.load(resolve_path("/data/processed/split/X_train.npy"))

X_train_descriptors = extract_sift_features(X_train_raw)

X_train_descriptors_stacked = np.vstack(X_train_descriptors)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_descriptors_stacked)

pca = PCA(n_components=60, random_state=42)
reduced_descriptors = pca.fit_transform(X_scaled)

print("Performing HDBSCAN clustering...")

model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
labels = model.fit_predict(reduced_descriptors)

print("Training set clustering completed...")

clusters_count = len(set(labels)) - (1 if -1 in labels else 0)
print(clusters_count)

train_histograms = []
for descriptors in X_train_descriptors:
    if descriptors is not None:
        words = model.fit_predict(descriptors)
        histogram, _ = np.histogram(words, bins=np.arange(clusters_count + 1))
    else:
        histogram = np.zeros(clusters_count)
    train_histograms.append(histogram)


test_histograms = []
for descriptors in X_train_descriptors:
    if descriptors is not None:
        words = model.fit_predict(descriptors)
        histogram, _ = np.histogram(words, bins=np.arange(clusters_count + 1))
    else:
        histogram = np.zeros(clusters_count)
    test_histograms.append(histogram)


train_hist_result = np.array(train_histograms, dtype=np.float32)
test_hist_result = np.array(test_histograms, dtype=np.float32)


train_labels = np.load(resolve_path("/data/processed/split/y_train.npy"))
test_labels = np.load(resolve_path("/data/processed/split/y_test.npy"))

print("Training Random Forest Classifier...")
random_forest_model = train_random_forest_classifier(train_histograms, train_labels)
os.makedirs(resolve_path("/models/ensemble/hdbscan"), exist_ok=True)
evaluate_model(random_forest_model, test_histograms, test_labels, resolve_path(f"/results/ensemble/hdbscan/results.txt"))
perform_k_fold_cross_validation(train_histograms, train_labels, n_splits=10, results_file_path=resolve_path(f"/results/ensemble/hdbscan/k-fold-res.txt"))