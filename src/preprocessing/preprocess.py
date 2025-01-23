import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.common.path_utils import resolve_path

def balance_dataset(X, y, image_data_generator_settings, data_bias_ratio=1.0, augmentation_factor=1.1):
    class_counts = np.bincount(y)
    max_count = max(class_counts)

    balanced_X, balanced_y = [], []
    datagen = ImageDataGenerator(**image_data_generator_settings)

    for class_label in np.unique(y):
        class_samples = X[y == class_label]
        class_size = len(class_samples)

        desired_augmented_count = 0

        if class_size < max_count:
            if data_bias_ratio == 0:
                desired_augmented_count = class_size * augmentation_factor
            else:
                desired_augmented_count = int(max_count * data_bias_ratio * augmentation_factor - class_size) 
        elif class_size== max_count:
            desired_augmented_count = int(max_count* augmentation_factor - class_size) 

        if desired_augmented_count > 0:
            augmented_samples = []
            for batch in datagen.flow(class_samples, batch_size=desired_augmented_count, shuffle=False):
                augmented_samples.extend(batch)
                if len(augmented_samples) >= desired_augmented_count:
                    break
            class_samples = np.vstack((class_samples, augmented_samples[:desired_augmented_count]))

        balanced_X.append(class_samples)
        balanced_y.extend([class_label] * (class_size + desired_augmented_count))

    return np.vstack(balanced_X), np.array(balanced_y, dtype=np.int32)

def visualize_distribution(y, labels, title, filename):
    class_counts = np.bincount(y)
    class_labels = [f"{list(labels.keys())[idx]} (Count: {count})" for idx, count in enumerate(class_counts)]
    plt.figure(figsize=(16, 16))
    plt.pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.savefig(filename)
    print(f"Saved pie chart: {filename}")


def normalize_data(folder_path, target_size=(128, 128)):
    X, y = [], []
    class_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    class_labels = {label: idx for idx, label in enumerate(class_dirs)}

    for label, idx in class_labels.items():
        class_path = os.path.join(folder_path, label)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, target_size) / 255.0
                X.append(img_resized)
                y.append(idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, class_labels