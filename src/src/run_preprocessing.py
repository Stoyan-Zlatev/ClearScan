import numpy as np
from data_processing.preprocess import load_images_and_labels
import os

# Ensure the processed folder exists
os.makedirs("./data/processed", exist_ok=True)

# Define paths
train_dir = "./data/raw/train"
test_dir = "./data/raw/test"

# Load and preprocess data
X_train, y_train, label_map = load_images_and_labels(train_dir)
X_test, y_test, _ = load_images_and_labels(test_dir)

# Save processed data
np.save("./data/processed/X_train.npy", X_train)
np.save("./data/processed/y_train.npy", y_train)
np.save("./data/processed/X_test.npy", X_test)
np.save("./data/processed/y_test.npy", y_test)

print("Data preprocessing complete!")
print(f"Label map: {label_map}")
