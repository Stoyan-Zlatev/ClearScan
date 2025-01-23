import os
from src.common.path_utils import resolve_path
from src.preprocessing.create_subsets import create_stratified_subsets
import numpy as np

if __name__ == "__main__":
    print("Loading data...")
    X= np.load(resolve_path("/data/processed/balanced/X.npy"))
    y = np.load(resolve_path("/data/processed/balanced/y.npy"))

    print("Creating subsets...")

    X_train, X_test, X_verify, y_train, y_test, y_verify = create_stratified_subsets(X, y, 0.8, 0.1, 0.1)

    print("Saving split datasets...")

    os.makedirs(resolve_path("/data/processed/split"), exist_ok=True)

    np.save(resolve_path("/data/processed/split/X_train.npy"), X_train)
    np.save(resolve_path("/data/processed/split/X_test.npy"), X_test)
    np.save(resolve_path("/data/processed/split/X_verify.npy"), X_verify)
    np.save(resolve_path("/data/processed/split/y_train.npy"), y_train)
    np.save(resolve_path("/data/processed/split/y_test.npy"), y_test)
    np.save(resolve_path("/data/processed/split/y_verify.npy"), y_verify)

    print("Split datasets saved.")