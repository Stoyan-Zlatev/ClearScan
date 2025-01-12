import multiprocessing

import numpy as np
from feature_extraction import extract_sift_features, create_visual_vocabulary, compute_histograms, find_optimal_k, \
    elbow_method

# Load processed data
X_train = np.load("../data/processed/X_train.npy")
X_val = np.load("../data/processed/X_val.npy")
X_test = np.load("../data/processed/X_test.npy")

print("Extracting SIFT features...")
train_descriptors = extract_sift_features(X_train, num_threads=multiprocessing.cpu_count() - 2)
#val_descriptors = extract_sift_features(X_val, num_threads=multiprocessing.cpu_count() - 2)
#test_descriptors = extract_sift_features(X_test, num_threads=multiprocessing.cpu_count() - 2)

print("Finding optimal k using silhouette analysis...")
k_range = range(10, 101, 25)  # Test k values from 10 to 100 in steps of 10
#optimal_k, scores = find_optimal_k(train_descriptors, k_range, num_threads=multiprocessing.cpu_count() - 2)
#print(f"Optimal k found: {optimal_k}")

print("Performing the elbow method...")
wcss = elbow_method(train_descriptors, k_range, num_threads=multiprocessing.cpu_count() - 4)

print("Elbow method completed. Review the plot to determine the optimal k.")

print("Creating visual vocabulary with optimal k...")
kmeans_model = create_visual_vocabulary(train_descriptors, k=optimal_k, num_threads=multiprocessing.cpu_count() - 2)

print("Converting images to histograms...")
X_train_hist = compute_histograms(train_descriptors, kmeans_model)
X_val_hist = compute_histograms(val_descriptors, kmeans_model)
X_test_hist = compute_histograms(test_descriptors, kmeans_model)

# Save histograms for further use
print("Saving histograms...")
np.save("../data/processed/X_train_hist.npy", X_train_hist)
np.save("../data/processed/X_val_hist.npy", X_val_hist)
np.save("../data/processed/X_test_hist.npy", X_test_hist)

print("Feature extraction and visual vocabulary creation complete!")
