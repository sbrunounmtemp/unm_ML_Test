#!/bin/python

# This script runs fraud detection experiments using our Random Forest and compares results with scikit-learn to validate correctness and efficiency.

import numpy as np
import pandas as pd
import sys
import os
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from project_setup import DataLoader
from random_forest import RandomForest

# Function to prepare the dataset for use in training and testing.
def prepare_data(loader):
    """Prepare features and target from the loader.

    Uses cached data when available. Performs lightweight preprocessing only
    (fillna, convert object/categorical cols to codes). Avoids re-encoding
    numeric columns or re-doing work that may be handled by `DataLoader`.
    """
    print("\n Preparing data...")

    # Use DataLoader's train_data directly
    df = loader.train_data
    if df is None:
        raise ValueError("Train data not loaded. Call loader.load_data() first.")

    # Determine feature columns (exclude IDs and target)
    feature_columns = [col for col in df.columns if col not in ['TransactionID', 'isFraud']]

    X = df[feature_columns].copy()
    y = df['isFraud'].values

    # Lightweight preprocessing: fill missing values
    X = X.fillna(0)

    # Use categorical features list from loader if present
    categorical_features = getattr(loader, 'categorical_features', [])

    # Convert only object or category dtypes to integer codes; skip numeric columns
    for col in categorical_features:
        if col in X.columns:
            if pd.api.types.is_categorical_dtype(X[col]):
                X[col] = X[col].cat.codes
            elif X[col].dtype == object:
                X[col] = pd.Categorical(X[col]).codes
            else:
                # numeric types left as-is (often already encoded)
                pass

    return X, y, categorical_features

# Function to run the full fraud detection: loads data, trains our Random Forest, compares with scikit-learn, and generates a submission file.
def main():
    print("="*70)
    print("FRAUD DETECTION")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_folder = os.path.join(project_root, "data")
    
    # Prefer parquet cache when available (DataLoader defaults to use_parquet=True)
    loader = DataLoader(data_folder, use_parquet=True)
    if not loader.load_data():
        return
    
    # Prepare data
    X, y, categorical_features = prepare_data(loader)
    
    print(f"Dataset: {X.shape}")
    print(f"Fraud rate: {np.mean(y)*100:.2f}%")
    
    # Use smaller subset for our implementation (to ensure it completes)
    print("\n*** Using 100,000 samples for our implementation ***")
    subset_size = 100000
    subset_indices = np.random.choice(len(X), subset_size, replace=False)
    X_subset = X.iloc[subset_indices]
    y_subset = y[subset_indices]
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_subset, y_subset, test_size=0.2, random_state=42, stratify=y_subset
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # OUR IMPLEMENTATION (with subset)
    print("\n" + "="*60)
    print("OUR RANDOM FOREST IMPLEMENTATION")
    print("="*60)
    
    our_rf = RandomForest(
        n_trees=10,  # Reduced trees
        max_depth=5,  
        min_samples_split=200,  # High minimum
        min_samples_leaf=100,  # High minimum
        max_features='sqrt',
        criterion='gini',
        chi_square_alpha=0.5,
        bootstrap=True,
        random_state=42
    )
    
    start = time.time()
    our_rf.fit(X_train, y_train, categorical_features=categorical_features)
    our_time = time.time() - start
    
    our_predictions = our_rf.predict(X_val)
    our_balanced_acc = calculate_balanced_accuracy(y_val, our_predictions)
    
    print(f"\n Time: {our_time:.2f} seconds")
    print(f"Balanced Accuracy: {our_balanced_acc:.4f}")
    
    # SKLEARN COMPARISON (with full data for better score)
    print("\n" + "="*60)
    print("SKLEARN RANDOM FOREST")
    print("="*60)
    
    # Use more data for sklearn
    X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Using full data: {len(X_train_full)} training samples")
    
    sklearn_rf = RandomForestClassifier(
        n_estimators=20,
        max_depth=6,
        min_samples_split=100,
        min_samples_leaf=50,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1 
    )
    
    start = time.time()
    sklearn_rf.fit(X_train_full, y_train_full)
    sklearn_time = time.time() - start
    
    sklearn_predictions = sklearn_rf.predict(X_val_full)
    sklearn_balanced_acc = calculate_balanced_accuracy(y_val_full, sklearn_predictions)
    
    print(f"\n Time: {sklearn_time:.2f} seconds")
    print(f"Balanced Accuracy: {sklearn_balanced_acc:.4f}")
    
    # CREATE SUBMISSION (using sklearn for reliability)
    print("\n" + "="*60)
    print("CREATING SUBMISSION FILE")
    print("="*60)
    
    # Prepare test data
    feature_columns = [col for col in loader.test_data.columns 
                      if col not in ['TransactionID']]
    X_test = loader.test_data[feature_columns].copy()
    test_ids = loader.test_data['TransactionID']
    
    # Preprocess test data
    X_test = X_test.fillna(0)
    for col in categorical_features:
        if col in X_test.columns:
            X_test[col] = pd.Categorical(X_test[col]).codes
    
    # Make predictions with sklearn model
    test_predictions = sklearn_rf.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud': test_predictions
    })
    
    submission_path = os.path.join(project_root, 'submission_random_forest_final.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission saved to: {submission_path}")
    print(f"Predicted {np.sum(test_predictions)} fraud cases out of {len(test_predictions)}")
    print(f"Fraud rate: {np.mean(test_predictions)*100:.2f}%")
    
    # SUMMARY FOR REPORT
    print("\n" + "="*70)
    print("SUMMARY FOR YOUR REPORT")
    print("="*70)
    
    print("\n1. Implementation Comparison:")
    print(f"   Our Implementation: {our_balanced_acc:.4f} (on 100k samples)")
    print(f"   Sklearn Reference: {sklearn_balanced_acc:.4f} (on full data)")
    
    print("\n2. Recommended Parameters (based on testing):")
    print("   - max_depth: 5-6")
    print("   - min_samples_split: 100-200") 
    print("   - min_samples_leaf: 50-100")
    print("   - n_trees: 10-20 for reasonable runtime")
    
    print("\n3. Class Imbalance:")
    print(f"   Original fraud rate: {np.mean(y)*100:.2f}%")
    print("   Strategy: Used balanced accuracy metric")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Function to compute balanced accuracy by averaging recall for fraud and normal classes to handle class imbalance.
def calculate_balanced_accuracy(y_true, y_pred):
    
    fraud_mask = y_true == 1
    normal_mask = y_true == 0
    
    recall_fraud = np.mean(y_pred[fraud_mask] == 1) if np.any(fraud_mask) else 0
    recall_normal = np.mean(y_pred[normal_mask] == 0) if np.any(normal_mask) else 0
    
    return (recall_fraud + recall_normal) / 2

if __name__ == "__main__":
    main()
