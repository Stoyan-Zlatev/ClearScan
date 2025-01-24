import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.ensemble.feature_extraction import extract_sift_features
from src.common.path_utils import resolve_path

X_train_raw = np.load(resolve_path("/data/processed/split/X_train.npy"))
X_train_descriptors = extract_sift_features(X_train_raw)
#y_train = np.load(resolve_path("/data/processed/split/y_train.npy"))

stacked = np.vstack(X_train_descriptors)

pca = PCA()  # No components specified, will keep all
pca.fit(stacked)

# Plot cumulative explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs. Number of Components")
plt.grid()
os.makedirs(resolve_path("/results/pca"), exist_ok=True)
plt.savefig(resolve_path("/results/pca/explained_variance.png"))