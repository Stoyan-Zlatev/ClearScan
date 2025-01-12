import numpy as np
from preprocess import load_and_preprocess_images, augment_images
import os

train_dir = "../data/raw/train"
test_dir = "../data/raw/test"
val_dir = "../data/raw/val"

os.makedirs("../data/processed", exist_ok=True)

print("Loading and preprocessing training data...")
X_train, y_train, label_map = load_and_preprocess_images(train_dir)
print("Training data shape:", X_train.shape, y_train.shape)

print("Loading and preprocessing validation data...")
X_val, y_val, _ = load_and_preprocess_images(val_dir)
print("Validation data shape:", X_val.shape, y_val.shape)

print("Loading and preprocessing test data...")
X_test, y_test, _ = load_and_preprocess_images(test_dir)
print("Test data shape:", X_test.shape, y_test.shape)

# Apply data augmentation carefully (set augmentation_count to a low number)
print("Applying data augmentation...")
#TODO augmentation raises memory issue
#TODO implement On-the-Fly Augmentation: Use tools like ImageDataGenerator in TensorFlow or mplement On-the-Fly Augmentation: Use tools like ImageDataGenerator in TensorFlow or torchvision.transforms in PyTorch to apply augmentation dynamically during training.torchvision.transforms in PyTorch to apply augmentation dynamically during training.
X_train_augmented, y_train_augmented = augment_images(X_train, y_train, batch_size=5, augmentation_count=0)
print("Augmented training data shape:", X_train_augmented.shape, y_train_augmented.shape)

# Save processed data
print("Saving processed arrays...")
np.save("../data/processed/X_train.npy", X_train_augmented)
np.save("../data/processed/y_train.npy", y_train_augmented)
np.save("../data/processed/X_val.npy", X_val)
np.save("../data/processed/y_val.npy", y_val)
np.save("../data/processed/X_test.npy", X_test)
np.save("../data/processed/y_test.npy", y_test)

print("Data preprocessing complete!")
print(f"Label map: {label_map}")
