from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score

from src.common.path_utils import get_timestamp, resolve_path


def find_optimal_k(descriptors, k_range, num_threads=None):
    """
    Perform silhouette analysis in parallel to find the optimal number of clusters.

    Args:
        descriptors: List of descriptors from SIFT.
        k_range: List or range of k values to evaluate.
        num_threads: Number of threads for parallel computation.

    Returns:
        best_k: Optimal k value based on silhouette score.
        silhouette_scores: List of silhouette scores for each k.
    """
    stacked_descriptors = np.vstack(descriptors)
    if num_threads is None:
        num_threads = multiprocessing.cpu_count() / 2

    def evaluate_k(k):
        print(f"Evaluating k={k}...")
        #kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, n_init=5)
        cluster_labels = kmeans.fit_predict(stacked_descriptors)
        score = silhouette_score(stacked_descriptors, cluster_labels)
        print(f"k={k} with Silhouette Score: {score}")
        return k, score
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(evaluate_k, k_range))

    silhouette_scores = [score for _, score in results]
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal k: {best_k} with Silhouette Score: {max(silhouette_scores)}")
    return best_k, silhouette_scores


# elbow_method_kmeans.py
def elbow_method(descriptors, k_range, num_threads=None):
    """
    Perform the elbow method to determine the optimal number of clusters for k-Means in parallel.

    Args:
        descriptors: List of descriptors from SIFT.
        k_range: List or range of k values to evaluate.
        num_threads: Number of threads for parallel computation.

    Returns:
        wcss: List of within-cluster sum of squares for each k.
    """
    #stacked_descriptors = np.vstack(descriptors)

    if num_threads is None:
        num_threads = multiprocessing.cpu_count()/2

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(descriptors)

    pca = PCA(n_components=60, random_state=42)
    reduced_descriptors = pca.fit_transform(X_scaled)

    def compute_wcss(k):
        print(f"Evaluating k={k}...")
        # Reduce to 50 dimensions (or another appropriate number)
        
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024*128, n_init=10)
        
        # n_init = 5
        #kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(reduced_descriptors)
        print(f"For k={k} wcss={kmeans.inertia_}")
        return kmeans.inertia_  # Inertia is the WCSS

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        wcss = list(executor.map(compute_wcss, k_range))

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.title("Elbow Method for Optimal k")
    plt.xticks(k_range)
    plt.grid()

    # Save the plot to a file instead of displaying it
    os.makedirs(resolve_path(f"/results/ensemble"), exist_ok=True)
    plt.savefig(resolve_path(f"/results/ensemble/{get_timestamp()}-elbow_method_plot.png"))
    print("Elbow plot saved!")

    return wcss