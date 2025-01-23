import os
from src.common.path_utils import resolve_path
from src.preprocessing.preprocess import normalize_data, visualize_distribution, balance_dataset
import numpy as np

if __name__ == "__main__":
    print("Loading data...")
    folder_path = resolve_path("/data/raw")

    print("Normalizing data...")
    X, y, labels = normalize_data(folder_path)

    print("Saving normalized data...")
    os.makedirs(resolve_path("/data/processed/normalized"), exist_ok=True)

    np.save(resolve_path("/data/processed/normalized/X.npy"), X)
    np.save(resolve_path("/data/processed/normalized/y.npy"), y)
    np.save(resolve_path("/data/processed/normalized/labels.npy"), labels)

    print("Normalized data saved.")

    print("Visualizing original distribution...")
    os.makedirs(resolve_path("/results/balanced"), exist_ok=True)
    visualize_distribution(y, labels, "Original Data Distribution", resolve_path("/results/balanced/original_distribution.png"))

    print("Balancing dataset...")
    X_balanced, y_balanced = balance_dataset(X, y, {"horizontal_flip":True, "rotation_range":15})

    print("Visualizing balanced distribution...")
    visualize_distribution(y_balanced, labels, "Balanced Data Distribution", resolve_path("/results/balanced/balanced_distribution.png"))

    os.makedirs(resolve_path("/data/processed/balanced"), exist_ok=True)

    np.save(resolve_path("/data/processed/balanced/X.npy"), X_balanced)
    np.save(resolve_path("/data/processed/balanced/y.npy"), y_balanced)
    print("Balanced dataset saved.")