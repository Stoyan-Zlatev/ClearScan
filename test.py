import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


from src.common.path_utils import resolve_path


model_stamp = "2025-01-23-11-19"

print("Loading balanced dataset...")
X_test = np.load(resolve_path("/data/processed/split/X_test.npy"))
y_test = np.load(resolve_path("/data/processed/split/y_test.npy"))

model = load_model(resolve_path(f"/models/cnn/{model_stamp}/model.keras"))

print("Preparing data...")

datagen = ImageDataGenerator()
test_gen = datagen.flow(X_test, y_test, batch_size=32)


results = model.evaluate(test_gen)
print(results)
test_loss, test_accuracy = results
report = f"Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.4f}\n\n"

y_pred = (model.predict(X_test) > 0.5).astype("int32")
report = report + "Classification Report:\n"
report = report + classification_report(y_test, y_pred)
report = report +"Confusion Matrix:\n"
report = report + str(confusion_matrix(y_test, y_pred))

print(report)

os.makedirs(resolve_path(f"/results/cnn/{model_stamp}"), exist_ok=True)
with open(resolve_path(f"/results/cnn/{model_stamp}/validation_report.txt"), "w") as f:
    f.write(report)