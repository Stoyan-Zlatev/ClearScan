import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Ensure a suitable backend
matplotlib.use("Agg")  # For saving plots as images without GUI

def plot_histogram(histogram, k, filename="single_histogram.png"):
    """
    Plot and save a single histogram of visual words.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(k), histogram, color='blue', alpha=0.7)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Histogram of Visual Words")
    plt.savefig(filename)  # Save the figure
    print(f"Saved histogram as {filename}")

def plot_multiple_histograms(histograms, num_images=5, filename_prefix="multiple_histograms"):
    """
    Plot and save histograms of multiple images in subplots.
    """
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.bar(range(histograms.shape[1]), histograms[i], color='blue', alpha=0.7)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title(f"Image {i+1}")
        plt.tight_layout()

    filename = f"{filename_prefix}.png"
    plt.savefig(filename)  # Save the figure
    print(f"Saved multiple histograms as {filename}")

if __name__ == "__main__":
    # Load the histograms
    print("Loading histograms...")
    X_train_hist = np.load("../data/processed/X_train_hist.npy")

    # Single histogram visualization
    print("Visualizing a single histogram...")
    plot_histogram(X_train_hist[0], k=X_train_hist.shape[1])

    # Multiple histogram visualization
    print("Visualizing multiple histograms...")
    plot_multiple_histograms(X_train_hist, num_images=5)

    print("Visualization complete!")
