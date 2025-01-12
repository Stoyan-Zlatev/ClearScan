import numpy as np
import matplotlib.pyplot as plt

def plot_class_distribution(y, labels, title, filename):
    class_counts = np.bincount(y)
    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=[f"{labels[i]} ({class_counts[i]})" for i in range(len(class_counts))],
            autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.savefig(filename)
    print(f"Saved pie chart: {filename}")

def plot_sample_images(X, y, labels, filename):
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(X[i])
        plt.title(f"Label: {labels[y[i]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved sample images plot: {filename}")

if __name__ == "__main__":
    print("Loading data...")
    X = np.load("../data/processed/X_train.npy")
    y = np.load("../data/processed/y_train.npy")

    labels = {0: "NORMAL", 1: "PNEUMONIA"}
    plot_class_distribution(y, labels, "Balanced Data Distribution", "balanced_data_distribution.png")
    plot_sample_images(X, y, labels, "sample_images.png")
