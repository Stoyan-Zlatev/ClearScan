import argparse
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from src.common.path_utils import resolve_path
from src.ensemble.train_random_forest import evaluate_model, perform_k_fold_cross_validation, train_random_forest_classifier

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train CNN model with validation")
    parser.add_argument('--perform_k_fold', type=bool, required=False, default=True, help="Perform k-fold cross-validation")
    parser.add_argument('--k_fold_n', type=int, required=False, default=10, help="Number of splits for k-fold cross-validation")
    parser.add_argument('--kmeans_k', type=int, required=False, default=20, help="Number of clusters for k-means histograms")
    args = parser.parse_args()

    print("Loading histogram features and labels...")
    
    train_histograms = np.load(resolve_path(f"/data/processed/split/X_train.npy"))
    #train_histograms = np.load(resolve_path(f"/data/processed/hist/{args.kmeans_k}/X_train.npy"))
    train_labels = np.load(resolve_path("/data/processed/split/y_train.npy"))

    test_histograms = np.load(resolve_path(f"/data/processed/split/X_test.npy"))
    #test_histograms = np.load(resolve_path(f"/data/processed/hist/{args.kmeans_k}/X_test.npy"))
    test_labels = np.load(resolve_path("/data/processed/split/y_test.npy"))

    # Define a parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, 40, 50, 100, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8, 16]
    }
    # Initialize the Random Forest model
    rf = RandomForestClassifier(random_state=42)

    print("Starting grid search...")

    # Grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(train_histograms, train_labels)

    # Best hyperparameters
    best_params = grid_search.best_params_

    print(str(best_params))

    print("Evaluating best model...")

    # Evaluate on test set
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(test_histograms)
    classification_rep = classification_report(test_labels, y_pred)

    print("Results...")
    print(str(best_params))
    print(str(classification_rep))

    print("Saving model...")
    os.makedirs(resolve_path("/models/ensemble/random_forest"), exist_ok=True)
    joblib.dump(grid_search.best_estimator_, resolve_path(f"/models/ensemble/random_forest/model_k_{args.kmeans_k}.joblib"))
    print("Model saved.")

    os.makedirs(resolve_path("/results/ensemble"), exist_ok=True)
    evaluate_model(grid_search.best_estimator_, test_histograms, test_labels, resolve_path(f"/results/ensemble/random_forest_k_{args.kmeans_k}.txt"))

    if args.perform_k_fold:
        print("Performing Cross-Validation...")
        perform_k_fold_cross_validation(train_histograms, train_labels, n_splits=args.k_fold_n, results_file_path=resolve_path(f"/results/ensemble/random_forest_k_{args.kmeans_k}_stratified_{args.k_fold_n}_fold.txt"))
        print("Cross-Validation completed.")