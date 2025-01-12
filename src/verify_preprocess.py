import numpy as np
import matplotlib.pyplot as plt

# Load processed data
X_train = np.load("./data/processed/X_train.npy")
y_train = np.load("./data/processed/y_train.npy")

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Sample labels:", y_train[:10])

# Visualize some images
for i in range(5):
    plt.imshow(X_train[i])
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
    plt.show()
