from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import multiprocessing
import matplotlib.pyplot as plt


# Parallel SIFT Feature Extraction
def extract_sift_features(images, num_threads=None):
    """
    Extract SIFT features in parallel.
    """

    def process_image(img):
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return descriptors

    sift = cv2.SIFT_create()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_image, images))

    # Filter out None results and stack descriptors
    all_descriptors = [desc for desc in results if desc is not None]
    return all_descriptors


# Parallel K-Means Clustering
def create_visual_vocabulary(descriptors, k=20, num_threads=None):
    """
    Perform k-Means clustering in parallel to create a visual vocabulary.
    """
    stacked_descriptors = np.vstack(descriptors)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    # Perform parallel clustering by splitting the data if necessary
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    chunks = np.array_split(stacked_descriptors, num_threads)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda chunk: kmeans.fit(chunk), chunks))

    kmeans.cluster_centers_ = np.vstack([result.cluster_centers_ for result in results])
    return kmeans


# Parallel Silhouette Analysis
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
        num_threads = multiprocessing.cpu_count()

    def evaluate_k(k):
        print(f"Evaluating k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(stacked_descriptors)
        return k, silhouette_score(stacked_descriptors, cluster_labels)

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
    stacked_descriptors = np.vstack(descriptors)

    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    def compute_wcss(k):
        print(f"Evaluating k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(stacked_descriptors)
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
    plt.savefig("elbow_method_plot.png")
    print("Elbow plot saved as 'elbow_method_plot.png'.")

    return wcss

def compute_histograms(images_descriptors, kmeans):
    """
    Convert images into histograms of visual words using the vocabulary.
    """
    histograms = []

    for descriptors in images_descriptors:
        if descriptors is not None:
            words = kmeans.predict(descriptors)
            histogram, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
        else:
            histogram = np.zeros(kmeans.n_clusters)

        histograms.append(histogram)

    return np.array(histograms, dtype=np.float32)
