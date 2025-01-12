import numpy as np
from test_random_forest import test_random_forest_classifier

if __name__ == "__main__":
    # Load histogram features and labels
    print("Loading histogram features and labels...")
    histograms = np.load("../data/processed/X_train_hist.npy")
    labels = np.load("../data/processed/y_train.npy")

    print("Testing Random Forest Classifier...")
    test_random_forest_classifier(histograms, labels)