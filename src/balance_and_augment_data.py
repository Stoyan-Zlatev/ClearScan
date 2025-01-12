import os
import numpy as np
import cv2
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(folder_path, target_size=(128, 128)):
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


def visualize_distribution(y, labels, title, filename):
    class_counts = np.bincount(y)
    class_labels = [f"{labels[idx]} ({count})" for idx, count in enumerate(class_counts)]
    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.savefig(filename)
    print(f"Saved pie chart: {filename}")


def balance_dataset(X, y, labels):
    class_counts = np.bincount(y)
    max_count = max(class_counts)

    balanced_X, balanced_y = [], []
    datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=15)

    for class_label in np.unique(y):
        class_samples = X[y == class_label]
        class_size = len(class_samples)

        if class_size < max_count:
            augmented_samples = []
            for batch in datagen.flow(class_samples, batch_size=max_count - class_size, shuffle=False):
                augmented_samples.extend(batch)
                if len(augmented_samples) >= max_count - class_size:
                    break
            class_samples = np.vstack((class_samples, augmented_samples[:max_count - class_size]))

        balanced_X.append(class_samples)
        balanced_y.extend([class_label] * max_count)

    return np.vstack(balanced_X), np.array(balanced_y, dtype=np.int32)