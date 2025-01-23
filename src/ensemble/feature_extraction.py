from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from sklearn.cluster import KMeans
import multiprocessing

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

    if num_threads is None:
        num_threads = multiprocessing.cpu_count() / 2

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_image, images))

    # Filter out None results and stack descriptors
    all_descriptors = [desc for desc in results if desc is not None]
    return all_descriptors


# Parallel K-Means Clustering
def create_visual_vocabulary(descriptors, k=20):
    """
    Perform k-Means clustering in parallel to create a visual vocabulary.
    """
    stacked_descriptors = np.vstack(descriptors)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(stacked_descriptors)
    return kmeans

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
