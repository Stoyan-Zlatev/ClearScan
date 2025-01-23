import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.common.path_utils import get_timestamp, resolve_path
from src.cnn.train_cnn import build_cnn, create_transfer_model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train CNN model with validation")
    parser.add_argument('--validate', type=bool, required=False, default=False, help="Specify whether to validate (True/False)")
    parser.add_argument('--epochs', type=int, required=False, default=20, help="Number of epochs to train the model")
    parser.add_argument('--batch_size', type=int, required=False, default=32, help="Batch size for training")
    parser.add_argument('--use_custom_model', type=bool, required=False, default=True, help="Use own custom model or not")
    args = parser.parse_args()

    timestamp = get_timestamp()

    print("Loading balanced dataset...")
    X_train = np.load(resolve_path("/data/processed/split/X_train.npy"))
    y_train = np.load(resolve_path("/data/processed/split/y_train.npy"))
    X_test = np.load(resolve_path("/data/processed/split/X_test.npy"))
    y_test = np.load(resolve_path("/data/processed/split/y_test.npy"))
    X_val = np.load(resolve_path("/data/processed/split/X_verify.npy"))
    y_val = np.load(resolve_path("/data/processed/split/y_verify.npy"))

    input_shape = X_train.shape[1:]

    model = None
    if args.use_custom_model:
        model = build_cnn(input_shape)
    else:
        model = create_transfer_model(input_shape)

    print("Preparing data...")
    datagen = ImageDataGenerator()
    train_gen = datagen.flow(X_train, y_train, batch_size=args.batch_size)
    val_gen = datagen.flow(X_val, y_val, batch_size=args.batch_size)
    test_gen = datagen.flow(X_test, y_test, batch_size=args.batch_size)

    print("Training model...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs)

    print("Saving model...")
    os.makedirs(resolve_path(f"/models/cnn/{timestamp}"), exist_ok=True)
    model.save(resolve_path(f"/models/cnn/{timestamp}/model.keras"))

    if(args.validate == False):
        exit(0)

    print("Evaluating model...")
   

    test_loss, test_accuracy = model.evaluate(test_gen)
    report = f"Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.4f}\n\n"

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    report = report + "Classification Report:\n"
    report = report + classification_report(y_test, y_pred)
    report = report +"Confusion Matrix:\n"
    report = report + str(confusion_matrix(y_test, y_pred))

    print(report)

    os.makedirs(resolve_path(f"/results/cnn/{timestamp}"), exist_ok=True)
    with open(resolve_path(f"/results/cnn/{timestamp}/validation_report.txt"), "w") as f:
        f.write(report)

    with open(resolve_path(f"/results/cnn/{timestamp}/config.py"), "w") as f:
        with open(resolve_path("/src/cnn/train_cnn.py"), "r") as src:
            f.write(src.read())

    with open(resolve_path(f"/results/cnn/{timestamp}/config_args.txt"), "w") as f:
        f.write(str(args))

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

    os.makedirs(resolve_path(f"/results/cnn/{timestamp}"), exist_ok=True)
    plt.savefig(resolve_path(f"/results/cnn/{timestamp}/training_validation_plots.png"))
    print("Saved training and validation plots.")

