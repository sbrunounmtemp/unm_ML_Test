#!/bin/python

# This script implements a Random Forest by combining multiple decision trees with bootstrapping and feature randomness for fraud detection.

import numpy as np
import pandas as pd
from collections import Counter
import sys
import os
from joblib import Parallel, delayed
import time

# Add the scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from decision_tree import DecisionTree
from information_gain import InformationGainMeasures

# Class for random forest implementation for fraud detection.
class RandomForest:
    
    # Function to initialize the Random Forest with parameters controlling tree depth, splitting rules, feature selection, and sampling strategy.
    def __init__(self,
                 n_trees=100,
                 max_depth=10,
                 min_samples_split=20,
                 min_samples_leaf=10,
                 max_features='sqrt',  
                 criterion='gini',
                 chi_square_alpha=0.05,
                 bootstrap=True,
                 random_state=42,
                 n_jobs=1):  # Number of parallel jobs
        
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.chi_square_alpha = chi_square_alpha
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Random forest properties
        self.trees = []
        self.feature_importances_ = None
        self.oob_score_ = None  # Out-of-bag score
        self.n_features_ = None
        
        # Set random seed
        np.random.seed(random_state)
    
    # Function to determine how many features to consider at each split based on the chosen maximum features.
    def _calculate_max_features(self, n_features):
        
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        else:
            return n_features
    
    # Function to generate a bootstrap sample of the dataset and identify out-of-bag instances for validation.
    def _bootstrap_sample(self, X, y):
        
        n_samples = len(X)
        
        if self.bootstrap:
            # Sample with replacement
            bootstrap_indices = np.random.choice(
                n_samples, 
                size=n_samples, 
                replace=True
            )
            
            # Out-of-bag samples
            oob_indices = np.array(
                list(set(range(n_samples)) - set(bootstrap_indices))
            )
            
            X_bootstrap = X.iloc[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
        else:
            # Use all data
            X_bootstrap = X
            y_bootstrap = y
            oob_indices = np.array([])
        
        return X_bootstrap, y_bootstrap, oob_indices
    
    # Function to train a single decision tree on a bootstrap sample with randomly selected features for the Random Forest.
    def _train_single_tree(self, tree_index, X, y, feature_names, categorical_features):
        
        # Set random seed for the tree
        np.random.seed(self.random_state + tree_index)
        
        # Bootstrap sample
        X_bootstrap, y_bootstrap, oob_indices = self._bootstrap_sample(X, y)
        
        # Random feature selection
        n_features = len(feature_names)
        max_features = self._calculate_max_features(n_features)
        selected_features = np.random.choice(
            feature_names,
            size=max_features,
            replace=False
        )
        
        # Select only the chosen features
        X_tree = X_bootstrap[selected_features]
        
        # Determine which categorical features are in the selection
        categorical_in_selection = [
            feat for feat in categorical_features 
            if feat in selected_features
        ]
        
        # Create and train the tree
        tree = DecisionTree(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            chi_square_alpha=self.chi_square_alpha
        )
        
        # Train quietly (suppress output for individual trees)
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            tree.fit(
                X_tree, 
                y_bootstrap,
                feature_names=list(selected_features),
                categorical_features=categorical_in_selection
            )
        
        return tree, list(selected_features), oob_indices
    
    # Function to train the Random Forest by building multiple decision trees on bootstrap samples and aggregating their results with Out-of-Bag (OOB) evaluation and feature importance analysis.
    def fit(self, X, y, categorical_features=None):
        
        # Convert to DataFrame if necessary
        if isinstance(X, np.ndarray):
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        else:
            feature_names = list(X.columns)
        
        self.n_features_ = len(feature_names)
        categorical_features = categorical_features or []
        
        # Convert y to numpy array
        y = np.array(y)
        
        print("\n" + "="*70)
        print("TRAINING RANDOM FOREST")
        print("="*70)
        print(f"Number of trees: {self.n_trees}")
        print(f"Maximum features per tree: {self.max_features} ({self._calculate_max_features(self.n_features_)} features)")
        print(f"Criterion: {self.criterion}")
        print(f"Maximum depth: {self.max_depth}")
        print(f"Bootstrap: {self.bootstrap}")
        print(f"Dataset shape: {X.shape}")
        print(f"Fraud rate: {np.mean(y)*100:.2f}%")
        
        start_time = time.time()
        
        # Store tree info
        self.trees = []
        self.tree_features = []
        self.oob_predictions = np.zeros((len(X), 2))  # Store OOB predictions
        self.oob_counts = np.zeros(len(X))  # Count OOB occurrences
        
        print("\n Training trees:")
        
        # Train trees 
        for i in range(self.n_trees):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{self.n_trees} trees trained...")
            
            tree, selected_features, oob_indices = self._train_single_tree(
                i, X, y, feature_names, categorical_features
            )
            
            self.trees.append(tree)
            self.tree_features.append(selected_features)
            
            # Make OOB predictions if we have OOB samples
            if len(oob_indices) > 0 and self.bootstrap:
                X_oob = X.iloc[oob_indices][selected_features]
                oob_pred = tree.predict(X_oob)
                
                for idx, oob_idx in enumerate(oob_indices):
                    self.oob_predictions[oob_idx, oob_pred[idx]] += 1
                    self.oob_counts[oob_idx] += 1
        
        # Calculate OOB score if using bootstrap
        if self.bootstrap:
            valid_oob = self.oob_counts > 0
            if np.any(valid_oob):
                oob_decisions = np.argmax(self.oob_predictions[valid_oob], axis=1)
                self.oob_score_ = np.mean(oob_decisions == y[valid_oob])
                print(f"\n OOB Score: {self.oob_score_:.4f}")
        
        # Calculate feature importances
        self._calculate_feature_importances(feature_names)
        
        training_time = time.time() - start_time
        print(f"\n Training completed in {training_time:.2f} seconds")
        print(f"Average time per tree: {training_time/self.n_trees:.3f} seconds")
        
        # Print top important features
        print("\n Top 5 Most Important Features:")
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importance_df.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Function to predict class labels by aggregating tree outputs and assigning the most probable class to each sample.
    def predict(self, X):
        
        # Get probability predictions
        probabilities = self.predict_proba(X)
        
        # Return class with highest probability
        return (probabilities >= 0.5).astype(int)
    
    # Function to estimate fraud probabilities by averaging predictions from all decision trees in the Random Forest.
    def predict_proba(self, X):
        
        if len(self.trees) == 0:
            raise ValueError("Forest has not been trained yet!")
        
        # Convert to DataFrame if necessary
        if isinstance(X, np.ndarray):
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        # Collect predictions from all trees
        all_predictions = np.zeros((len(X), self.n_trees))
        
        for tree_idx, (tree, features) in enumerate(zip(self.trees, self.tree_features)):
            # Select features used by this tree
            X_tree = X[features]
            
            # Get predictions
            tree_predictions = tree.predict(X_tree)
            all_predictions[:, tree_idx] = tree_predictions
        
        # Average predictions (probability of fraud)
        fraud_probability = np.mean(all_predictions, axis=1)
        
        return fraud_probability
    
    # Function to compute the classification accuracy by comparing predicted labels with the true labels.
    def calculate_accuracy(self, X, y):
        
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    # Function to calculate balanced accuracy by averaging class-specific recalls to account for data imbalance.
    def calculate_balanced_accuracy(self, X, y):
        
        predictions = self.predict(X)
        
        # Calculate recall for each class
        fraud_mask = y == 1
        normal_mask = y == 0
        
        recall_fraud = np.mean(predictions[fraud_mask] == 1) if np.any(fraud_mask) else 0
        recall_normal = np.mean(predictions[normal_mask] == 0) if np.any(normal_mask) else 0
        
        # Balanced accuracy is average of recalls
        balanced_accuracy = (recall_fraud + recall_normal) / 2
        
        return balanced_accuracy
    
    # Function to compute feature importance scores based on their frequency of use across trees in the Random Forest.
    def _calculate_feature_importances(self, feature_names):
        
        importance_counts = {feature: 0 for feature in feature_names}
        
        # Count how often each feature is used in trees
        for features in self.tree_features:
            for feature in features:
                importance_counts[feature] += 1
        
        # Normalize by number of trees
        self.feature_importances_ = np.array([
            importance_counts[feature] / self.n_trees 
            for feature in feature_names
        ])
    
    # Function to retrieve overall Random Forest statistics, including tree depths, node counts, leaf counts, and OOB score.
    def get_forest_stats(self):
        
        tree_depths = []
        tree_nodes = []
        tree_leaves = []
        
        for tree in self.trees:
            stats = tree.get_tree_stats()
            tree_depths.append(stats.get('max_depth', 0))
            tree_nodes.append(stats.get('total_nodes', 0))
            tree_leaves.append(stats.get('total_leaves', 0))
        
        return {
            'n_trees': self.n_trees,
            'avg_tree_depth': np.mean(tree_depths),
            'max_tree_depth': np.max(tree_depths),
            'min_tree_depth': np.min(tree_depths),
            'avg_tree_nodes': np.mean(tree_nodes),
            'avg_tree_leaves': np.mean(tree_leaves),
            'oob_score': self.oob_score_
        }

# Function to test the Random Forest on synthetic fraud data by comparing its performance to a single decision tree and display the results.
def test_random_forest():
    
    print("\n" + "="*70)
    print("TESTING RANDOM FOREST IMPLEMENTATION")
    print("="*70)
    
    # Create synthetic fraud detection data
    np.random.seed(42)
    n_samples = 2000
    
    # Generate features
    X = pd.DataFrame({
        'transaction_amount': np.random.exponential(50, n_samples),
        'days_since_last': np.random.randint(0, 30, n_samples),
        'num_transactions': np.random.poisson(3, n_samples),
        'merchant_risk': np.random.uniform(0, 1, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'account_age': np.random.randint(0, 365, n_samples),
        'previous_failures': np.random.poisson(0.5, n_samples)
    })
    
    # Generate labels with some patterns
    fraud_probability = (
        (X['transaction_amount'] > 200) * 0.3 +
        (X['days_since_last'] < 1) * 0.2 +
        (X['num_transactions'] > 5) * 0.2 +
        (X['merchant_risk'] > 0.7) * 0.3 +
        (X['previous_failures'] > 1) * 0.2
    )
    y = (np.random.random(n_samples) < fraud_probability).astype(int)
    
    print(f"Dataset: {n_samples} samples, {np.sum(y)} fraud cases ({np.mean(y)*100:.1f}%)")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Compare single tree vs Random Forest
    print("\n" + "="*60)
    print("COMPARISON: Single Tree vs Random Forest")
    print("="*60)
    
    # Train single decision tree
    print("\n1. Single Decision Tree:")
    single_tree = DecisionTree(
        criterion='gini',
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        chi_square_alpha=0.05
    )
    single_tree.fit(X_train, y_train)
    
    single_acc = single_tree.calculate_accuracy(X_test, y_test)
    single_balanced = single_tree.calculate_balanced_accuracy(X_test, y_test)
    
    print(f"  Test Accuracy: {single_acc:.4f}")
    print(f"  Test Balanced Accuracy: {single_balanced:.4f}")
    
    # Train Random Forest
    print("\n2. Random Forest (10 trees):")
    rf_small = RandomForest(
        n_trees=10,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        criterion='gini',
        chi_square_alpha=0.05,
        bootstrap=True,
        random_state=42
    )
    rf_small.fit(X_train, y_train)
    
    rf_acc = rf_small.calculate_accuracy(X_test, y_test)
    rf_balanced = rf_small.calculate_balanced_accuracy(X_test, y_test)
    
    print(f"\n  Test Accuracy: {rf_acc:.4f}")
    print(f"  Test Balanced Accuracy: {rf_balanced:.4f}")
    print(f"  Improvement over single tree: {(rf_balanced - single_balanced)*100:.1f}%")
    
    # Get forest statistics
    forest_stats = rf_small.get_forest_stats()
    print(f"\n  Forest Statistics:")
    print(f"    Average tree depth: {forest_stats['avg_tree_depth']:.1f}")
    print(f"    Average nodes per tree: {forest_stats['avg_tree_nodes']:.1f}")
    
    print("\n Random Forest implementation complete and tested.")
    

if __name__ == "__main__":
    test_random_forest()
