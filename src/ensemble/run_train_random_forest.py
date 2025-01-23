import argparse
import os
import joblib
import numpy as np
from src.common.path_utils import resolve_path
from src.ensemble.train_random_forest import evaluate_model, perform_k_fold_cross_validation, train_random_forest_classifier

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train CNN model with validation")
    parser.add_argument('--perform_k_fold', type=bool, required=False, default=True, help="Perform k-fold cross-validation")
    parser.add_argument('--k_fold_n', type=int, required=False, default=20, help="Number of splits for k-fold cross-validation")
    parser.add_argument('--kmeans_k', type=int, required=False, default=20, help="Number of clusters for k-means histograms")
    args = parser.parse_args()

    print("Loading histogram features and labels...")
    train_histograms = np.load(resolve_path(f"/data/processed/hist/{args.kmeans_k}/X_train.npy"))
    train_labels = np.load(resolve_path("/data/processed/split/y_train.npy"))

    test_histograms = np.load(resolve_path(f"/data/processed/hist/{args.kmeans_k}/X_test.npy"))
    test_labels = np.load(resolve_path("/data/processed/split/y_test.npy"))

    print("Training Random Forest Classifier...")
    random_forest_model = train_random_forest_classifier(train_histograms, train_labels)

    print("Saving model...")
    os.makedirs(resolve_path("/models/ensemble/random_forest"), exist_ok=True)
    joblib.dump(random_forest_model, resolve_path(f"/models/ensemble/random_forest/model_k_{args.kmeans_k}.joblib"))
    print("Model saved.")

    os.makedirs(resolve_path("/results/ensemble"), exist_ok=True)
    evaluate_model(random_forest_model, test_histograms, test_labels, resolve_path(f"/results/ensemble/random_forest_k_{args.kmeans_k}.txt"))

    if args.perform_k_fold:
        print("Performing Cross-Validation...")
        perform_k_fold_cross_validation(train_histograms, train_labels, n_splits=args.k_fold_n, results_file_path=resolve_path(f"/results/ensemble/random_forest_k_{args.kmeans_k}_stratified_{args.k_fold_n}_fold.txt"))
        print("Cross-Validation completed.")