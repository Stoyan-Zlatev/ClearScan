


#print("Finding optimal k using silhouette analysis...")
#k_range = range(10, 101, 25)  # Test k values from 10 to 100 in steps of 10
#optimal_k, scores = find_optimal_k(train_descriptors, k_range, num_threads=multiprocessing.cpu_count() - 2)
#print(f"Optimal k found: {optimal_k}")

import numpy as np
from src.ensemble.feature_extraction import extract_sift_features
from src.common.path_utils import resolve_path
from src.ensemble.k_find import elbow_method, find_optimal_k

k_range = range(1, 50, 1)

print("Loading data...")

X_train = np.load(resolve_path("/data/processed/split/X_train.npy"))

print("Extracting SIFT features...")
train_descriptors = extract_sift_features(X_train)

#print("Performing the elbow method...")
#wcss = elbow_method(train_descriptors, k_range)
#print("Elbow method completed. Review the plot to determine the optimal k.")

print("Performing Silhouette analysis...")
best_k, scores = find_optimal_k(train_descriptors, k_range)
print(f"Optimal k found: {best_k}")
