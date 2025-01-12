import numpy as np
from train_random_forest import train_random_forest_classifier

if __name__ == "__main__":
    # Load histogram features and labels
    print("Loading histogram features and labels...")
    histograms = np.load("../data/processed/X_train_hist.npy")
    labels = np.load("../data/processed/y_train.npy")

    print("Training Random Forest Classifier...")
    random_forest_model = train_random_forest_classifier(histograms, labels)

    print("Training complete. Model is ready for use.")