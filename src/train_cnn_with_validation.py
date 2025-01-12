import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

def build_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Loading balanced dataset...")
    X_train = np.load("../data/processed/X_train.npy")
    y_train = np.load("../data/processed/y_train.npy")
    X_val = np.load("../data/processed/X_val.npy")
    y_val = np.load("../data/processed/y_val.npy")

    input_shape = X_train.shape[1:]
    model = build_cnn(input_shape)

    print("Preparing data augmentation...")
    datagen = ImageDataGenerator()
    train_gen = datagen.flow(X_train, y_train, batch_size=32)
    val_gen = datagen.flow(X_val, y_val, batch_size=32)

    print("Training model...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=20)

    print("Evaluating model...")
    val_loss, val_accuracy = model.evaluate(val_gen)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    y_pred = (model.predict(X_val) > 0.5).astype("int32")
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("Saving model...")
    model.save("../models/cnn_with_validation.h5")

    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Loss")

    plt.savefig("../results/training_validation_plots.png")
    print("Saved training and validation plots.")
