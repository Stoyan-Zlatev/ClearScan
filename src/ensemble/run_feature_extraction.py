import argparse
import os
import joblib
import numpy as np
from src.common.path_utils import resolve_path
from src.ensemble.feature_extraction import extract_sift_features, create_visual_vocabulary, compute_histograms

parser = argparse.ArgumentParser(description="Extract SIFT features and create visual vocabulary")
parser.add_argument('--k', type=int, required=False, default=20, help="Number of clusters for k-means")
args = parser.parse_args()

print("Loading data...")
X_train = np.load(resolve_path("/data/processed/split/X_train.npy"))
X_test = np.load(resolve_path("/data/processed/split/X_test.npy"))
X_val = np.load(resolve_path("/data/processed/split/X_verify.npy"))

print("Extracting SIFT features...")
train_descriptors = extract_sift_features(X_train)
test_descriptors = extract_sift_features(X_test)
val_descriptors = extract_sift_features(X_val)

k = args.k

print("Creating visual vocabulary...")
kmeans_model = create_visual_vocabulary(train_descriptors, k=k)

os.makedirs(resolve_path("/models/ensemble/kmeans"), exist_ok=True)
joblib.dump(kmeans_model, resolve_path(f"/models/ensemble/kmeans/k_means_k_{k}.joblib"))

print("Converting images to histograms...")
X_train_hist = compute_histograms(train_descriptors, kmeans_model)
X_test_hist = compute_histograms(test_descriptors, kmeans_model)
X_val_hist = compute_histograms(val_descriptors, kmeans_model)

print("Saving histograms...")
os.makedirs(resolve_path(f"/data/processed/hist/{k}"), exist_ok=True)
np.save(resolve_path(f"/data/processed/hist/{k}/X_train.npy"), X_train_hist)
np.save(resolve_path(f"/data/processed/hist/{k}/X_verify.npy"), X_val_hist)
np.save(resolve_path(f"/data/processed/hist/{k}/X_test.npy"), X_test_hist)

print("Feature extraction and visual vocabulary creation completed!")
