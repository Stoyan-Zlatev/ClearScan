import os
import cv2
import numpy as np

def preprocess_image(img_path, target_size=(128, 128)):
    """
    Preprocesses a single image:
    - Converts to grayscale.
    - Resizes the image.
    - Normalizes pixel values.

    Args:
        img_path (str): Path to the image.
        target_size (tuple): Desired image size.

    Returns:
        np.array: Preprocessed image.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized / 255.0
    return img_normalized


def load_images_and_labels(folder_path, target_size=(128, 128)):
    """
    Loads and preprocesses images from a folder and returns the data and labels.

    Args:
        folder_path (str): Path to the folder containing class subfolders.
        target_size (tuple): Target image size.

    Returns:
        X (list): List of preprocessed images.
        y (list): List of labels.
        label_map (dict): Mapping of class names to numeric labels.
    """
    X, y = [], []
    label_map = {label: idx for idx, label in enumerate(os.listdir(folder_path))}

    for label, idx in label_map.items():
        class_folder = os.path.join(folder_path, label)
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img = preprocess_image(img_path, target_size)
            if img is not None:
                X.append(img)
                y.append(idx)

    return np.array(X), np.array(y), label_map
