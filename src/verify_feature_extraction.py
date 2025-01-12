import numpy as np

# Load histograms
X_train_hist = np.load("../data/processed/X_train_hist.npy")
X_val_hist = np.load("../data/processed/X_val_hist.npy")
X_test_hist = np.load("../data/processed/X_test_hist.npy")

print("Histogram shapes:")
print("Training set:", X_train_hist.shape)
print("Validation set:", X_val_hist.shape)
print("Test set:", X_test_hist.shape)

print("Sample histogram from training set:")
print(X_train_hist[0])
