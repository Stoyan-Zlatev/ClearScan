import os
import cv2
import numpy as np


def load_and_preprocess_images(folder_path, target_size=(128, 128)):
    """
    Load and preprocess images for a machine learning pipeline.
    """
    X, y = [], []
    # Filter out hidden files/folders
    class_names = [d for d in os.listdir(folder_path) if
                   not d.startswith('.') and os.path.isdir(os.path.join(folder_path, d))]
    label_map = {label: idx for idx, label in enumerate(class_names)}

    for label, idx in label_map.items():
        class_folder = os.path.join(folder_path, label)
        img_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_name in img_files:
            img_path = os.path.join(class_folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize and normalize the image
                img_resized = cv2.resize(img, target_size)
                img_normalized = img_resized.astype(np.float32) / 255.0
                X.append(img_normalized)
                y.append(idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, label_map


def augment_images(X, y, batch_size=100, augmentation_count=0):
    """
    Augment the dataset in batches to reduce memory usage.
    """
    X_augmented, y_augmented = [], []
    num_batches = (len(X) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(X))
        batch_X = X[start:end]
        batch_y = y[start:end]

        for img, label in zip(batch_X, batch_y):
            X_augmented.append(img)  # original image
            y_augmented.append(label)
            for _ in range(augmentation_count):
                augmented_img = apply_random_transform(img)
                X_augmented.append(augmented_img)
                y_augmented.append(label)

        # If memory is an issue, you can periodically convert and clear
        # but for now, we just keep them in memory.

    return np.array(X_augmented, dtype=np.float32), np.array(y_augmented, dtype=np.int32)


def apply_random_transform(img):
    """
    Apply a random transformation to the input image.
    """
    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    # Random rotation
    angle = np.random.uniform(-15, 15)
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

    return img
