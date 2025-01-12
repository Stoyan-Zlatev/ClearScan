from balance_and_augment_data import load_data, visualize_distribution, balance_dataset
import numpy as np

if __name__ == "__main__":
    print("Loading data...")
    folder_path = "../data/raw/train"
    X, y, labels = load_data(folder_path)

    print("Visualizing original distribution...")
    visualize_distribution(y, labels, "Original Data Distribution", "original_distribution.png")

    print("Balancing dataset...")
    X_balanced, y_balanced = balance_dataset(X, y, labels)

    print("Visualizing balanced distribution...")
    visualize_distribution(y_balanced, labels, "Balanced Data Distribution", "balanced_distribution.png")

    np.save("../data/processed/X_balanced.npy", X_balanced)
    np.save("../data/processed/y_balanced.npy", y_balanced)
    print("Balanced dataset saved.")