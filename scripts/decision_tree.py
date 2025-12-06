#!/bin/python

# This script implements a complete decision tree using information gain for splitting and chi-square test for stopping, with methods for training, prediction, and evaluation.

import numpy as np
import pandas as pd
from collections import Counter
import sys
import os

# Add the scripts directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from information_gain import InformationGainMeasures
from chi_square_test import ChiSquareTest


# Class that represents a singl node in the decision tree.
class TreeNode:
    

    # Function to initialize a tree node with properties for splits, predictions, child links, and node statistics.
    def __init__(self):
        # Node properties
        self.is_leaf = False
        self.prediction = None  # For leaf nodes: predicted class
        self.prediction_probability = None  # Probability of predicted class
        
        # Split properties (for internal nodes)
        self.split_feature = None  # Which feature to split on
        self.split_value = None  # Value to split at (for numerical features)
        self.split_categories = None  # Categories for left split (for categorical)
        
        # Child nodes
        self.left_child = None
        self.right_child = None
        
        # Node statistics
        self.n_samples = 0
        self.n_fraud = 0
        self.depth = 0
        self.impurity = 0
        
    # Function to return a readable string summary of the node, showing its type, prediction, and key statistics for debugging and analysis.
    def __str__(self):
        if self.is_leaf:
            return f"Leaf(pred={self.prediction}, prob={self.prediction_probability:.3f}, samples={self.n_samples})"
        else:
            return f"Node(feature={self.split_feature}, samples={self.n_samples}, fraud_rate={self.n_fraud/self.n_samples:.3f})"

# Class for complete decision tree implementation with Information Gain and Chi-Square stopping.
class DecisionTree:

    # Function to initialize the decision tree with splitting criteria, stopping rules, and supporting statistical tests.
    def __init__(self, 
                 criterion='entropy',  
                 max_depth=10,
                 min_samples_split=20,
                 min_samples_leaf=10,
                 chi_square_alpha=0.05):
        
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.chi_square_alpha = chi_square_alpha
        
        # Initialize helper classes
        self.ig_calculator = InformationGainMeasures()
        self.chi_square_tester = ChiSquareTest()
        
        # Tree properties
        self.root = None
        self.n_nodes = 0
        self.n_leaves = 0
        self.feature_names = None
        self.categorical_features = []
        
    
    # Function to train the decision tree by recursively building nodes from the input features and target labels using the specified criteria and stopping rules.
    def fit(self, X, y, feature_names=None, categorical_features=None):
        
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        self.feature_names = X.columns.tolist()
        self.categorical_features = categorical_features or []
        
        # Convert y to numpy array
        y = np.array(y)
        
        print(f"\n {'='*60}")
        print(f"Training Decision Tree")
        print(f"{'='*60}")
        print(f"Criterion: {self.criterion}")
        print(f"Max depth: {self.max_depth}")
        print(f"Min samples split: {self.min_samples_split}")
        print(f"Min samples leaf: {self.min_samples_leaf}")
        print(f"Chi-square alpha: {self.chi_square_alpha}")
        print(f"Dataset shape: {X.shape}")
        print(f"Fraud rate: {np.mean(y)*100:.2f}%")
        
        # Build the tree recursively
        self.root = self._build_tree(X, y, depth=0)
        
        print(f"\n Tree built successfully!")
        print(f"Total nodes: {self.n_nodes}")
        print(f"Total leaves: {self.n_leaves}")
        
    
    # Function to recursively construct the decision tree by selecting splits, applying stopping rules, and creating child or leaf nodes.
    def _build_tree(self, X, y, depth):
        
        node = TreeNode()
        node.depth = depth
        node.n_samples = len(y)
        node.n_fraud = np.sum(y)
        
        # Calculate node impurity
        if self.criterion == 'entropy':
            node.impurity = self.ig_calculator.calculate_entropy(y)
        elif self.criterion == 'gini':
            node.impurity = self.ig_calculator.calculate_gini_index(y)
        else:
            node.impurity = self.ig_calculator.calculate_misclassification_error(y)
        
        self.n_nodes += 1
        
        # Check stopping criteria
        if self._should_stop(X, y, depth):
            # Create leaf node
            node.is_leaf = True
            node.prediction = int(np.mean(y) >= 0.5)  # Predict 1 if majority fraud
            node.prediction_probability = np.mean(y == node.prediction)
            self.n_leaves += 1
            return node
        
        # Find best split
        best_feature, best_value, best_gain = self._find_best_split(X, y)
        
        if best_feature is None:
            # No valid split found, create leaf
            node.is_leaf = True
            node.prediction = int(np.mean(y) >= 0.5)
            node.prediction_probability = np.mean(y == node.prediction)
            self.n_leaves += 1
            return node
        
        # Make the split
        left_mask, right_mask = self._make_split(X, best_feature, best_value)
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # Check chi-square test
        should_stop, chi_info = self.chi_square_tester.should_stop_splitting(
            y, y_left, y_right, alpha=self.chi_square_alpha
        )
        
        if should_stop:
            # Chi-square test shows split is not significant
            node.is_leaf = True
            node.prediction = int(np.mean(y) >= 0.5)
            node.prediction_probability = np.mean(y == node.prediction)
            self.n_leaves += 1
            if depth < 3:  # Only print for top levels to avoid clutter
                print(f"  {'  '*depth}Chi-square stopped split at depth {depth} (p-value: {chi_info.get('p_value', 0):.4f})")
            return node
        
        # Create split node
        node.split_feature = best_feature
        node.split_value = best_value
        
        if depth < 3:  # Print top-level splits
            print(f"  {'  '*depth}Depth {depth}: Split on {best_feature} "
                  f"(gain={best_gain:.4f}, samples={node.n_samples}, "
                  f"fraud_rate={node.n_fraud/node.n_samples:.3f})")
        
        # Recursively build child nodes
        node.left_child = self._build_tree(X_left, y_left, depth + 1)
        node.right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return node
    
    # Function to determine whether a node should stop splitting based on depth, sample size, or class purity criteria.
    def _should_stop(self, X, y, depth):
        
        # Check max depth
        if depth >= self.max_depth:
            return True
        
        # Check minimum samples
        if len(y) < self.min_samples_split:
            return True
        
        # Check if node is pure (all same class)
        if len(np.unique(y)) == 1:
            return True
        
        return False
    
    # Function to identify the feature and threshold that provide the highest information gain for splitting the node.
    def _find_best_split(self, X, y):
        
        best_feature = None
        best_value = None
        best_gain = -1
        
        for feature in X.columns:
            # Skip if feature has only one unique value
            if X[feature].nunique() <= 1:
                continue
            
            # Handle categorical vs numerical features
            if feature in self.categorical_features:
                # For categorical features
                gain, split_value = self._find_best_categorical_split(X[feature], y)
            else:
                # For numerical features
                gain, split_value = self._find_best_numerical_split(X[feature], y)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_value = split_value
        
        return best_feature, best_value, best_gain
    
    # Function to determine the optimal threshold for a numerical feature by testing split points and selecting the one with maximum information gain.
    def _find_best_numerical_split(self, feature_values, y):
        
        best_gain = -1
        best_threshold = None
        
        # Get unique values and sort them
        unique_values = np.unique(feature_values)
        
        # Try different thresholds (midpoints between consecutive values)
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            
            # Split data
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            # Skip if split creates too small leaves
            if (np.sum(left_mask) < self.min_samples_leaf or 
                np.sum(right_mask) < self.min_samples_leaf):
                continue
            
            # Calculate information gain
            gain = self.ig_calculator.calculate_information_gain(
                y, y[left_mask], y[right_mask], self.criterion
            )
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        return best_gain, best_threshold
    
    #Function to find the best category-based split by testing partitions and selecting the option with maximum information gain.
    def _find_best_categorical_split(self, feature_values, y):
       
        best_gain = -1
        best_categories = None
        
        unique_values = feature_values.unique()
        
        # Try each category as left split
        for category in unique_values:
            left_mask = feature_values == category
            right_mask = ~left_mask
            
            # Skip if split creates too small leaves
            if (np.sum(left_mask) < self.min_samples_leaf or 
                np.sum(right_mask) < self.min_samples_leaf):
                continue
            
            # Calculate information gain
            gain = self.ig_calculator.calculate_information_gain(
                y, y[left_mask], y[right_mask], self.criterion
            )
            
            if gain > best_gain:
                best_gain = gain
                best_categories = {category}
        
        return best_gain, best_categories
    
    # Function to generate boolean masks to divide the dataset into left and right child nodes based on a feature split.
    def _make_split(self, X, feature, value):
        
        if feature in self.categorical_features:
            # Categorical split
            left_mask = X[feature].isin(value)
        else:
            # Numerical split
            left_mask = X[feature] <= value
        
        right_mask = ~left_mask
        return left_mask, right_mask
    
    # Function to predict class labels for input samples by traversing the trained decision tree.
    def predict(self, X):
        
        if self.root is None:
            raise ValueError("Tree has not been trained yet!")
        
        # Convert to DataFrame if necessary
        if isinstance(X, np.ndarray):
            if self.feature_names is None:
                raise ValueError("Feature names are required for numpy array input")
            X = pd.DataFrame(X, columns=self.feature_names)
        
        predictions = []
        for idx in range(len(X)):
            sample = X.iloc[idx]
            prediction = self._predict_sample(sample, self.root)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    # Function to estimate the probability of fraud (class 1) for each sample by traversing the trained decision tree.
    def predict_proba(self, X):
        
        if self.root is None:
            raise ValueError("Tree has not been trained yet!")
        
        # Convert to DataFrame if necessary
        if isinstance(X, np.ndarray):
            if self.feature_names is None:
                raise ValueError("Feature names are required for numpy array input")
            X = pd.DataFrame(X, columns=self.feature_names)
        
        probabilities = []
        for idx in range(len(X)):
            sample = X.iloc[idx]
            prob = self._predict_sample_proba(sample, self.root)
            probabilities.append(prob)
        
        return np.array(probabilities)
    
    # Function to recursively predict the class of a single sample by following decision rules down the tree.
    def _predict_sample(self, sample, node):
        
        if node.is_leaf:
            return node.prediction
        
        # Decide which child to go to
        if node.split_feature in self.categorical_features:
            # Categorical feature
            if sample[node.split_feature] in node.split_value:
                return self._predict_sample(sample, node.left_child)
            else:
                return self._predict_sample(sample, node.right_child)
        else:
            # Numerical feature
            if sample[node.split_feature] <= node.split_value:
                return self._predict_sample(sample, node.left_child)
            else:
                return self._predict_sample(sample, node.right_child)
    
    # Function to recursively estimate the fraud probability for a single sample by traversing the decision tree.
    def _predict_sample_proba(self, sample, node):
        
        if node.is_leaf:
            # Return fraud probability at this leaf
            return node.n_fraud / node.n_samples if node.n_samples > 0 else 0
        
        # Decide which child to go to
        if node.split_feature in self.categorical_features:
            if sample[node.split_feature] in node.split_value:
                return self._predict_sample_proba(sample, node.left_child)
            else:
                return self._predict_sample_proba(sample, node.right_child)
        else:
            if sample[node.split_feature] <= node.split_value:
                return self._predict_sample_proba(sample, node.left_child)
            else:
                return self._predict_sample_proba(sample, node.right_child)
    
    # Function to compute the accuracy score by comparing predicted labels with the true labels.
    def calculate_accuracy(self, X, y):
        
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    # Function to calculate balanced accuracy across classes to handle imbalanced data.
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
    
    # Function to retrieve key statistics of the decision tree, including depth, sample counts, and fraud rates.
    def get_tree_stats(self):
        
        if self.root is None:
            return {}
        
        depths = []
        fraud_rates = []
        sample_counts = []
        
        # Funtion to traverse the tree to collect node depths, sample counts, and fraud rates for computing overall statistics.
        def traverse(node):
            if node is None:
                return
            
            if node.is_leaf:
                depths.append(node.depth)
                fraud_rates.append(node.n_fraud / node.n_samples if node.n_samples > 0 else 0)
                sample_counts.append(node.n_samples)
            else:
                traverse(node.left_child)
                traverse(node.right_child)
        
        traverse(self.root)
        
        stats = {
            'total_nodes': self.n_nodes,
            'total_leaves': self.n_leaves,
            'max_depth': max(depths) if depths else 0,
            'avg_depth': np.mean(depths) if depths else 0,
            'min_leaf_samples': min(sample_counts) if sample_counts else 0,
            'max_leaf_samples': max(sample_counts) if sample_counts else 0,
            'avg_leaf_fraud_rate': np.mean(fraud_rates) if fraud_rates else 0
        }
        
        return stats
    
    # Function to display the tree structure up to a specified depth for visualization and interpretation.
    def print_tree_structure(self, max_depth=3):

        # Function to recursively print each node of the tree with predictions or split conditions for structural visualization.
        def print_node(node, depth=0, prefix="Root"):
            if node is None or depth > max_depth:
                return
            
            indent = "  " * depth
            
            if node.is_leaf:
                print(f"{indent}{prefix}: Leaf - Predict {node.prediction} "
                      f"(samples={node.n_samples}, fraud_rate={node.n_fraud/node.n_samples:.3f})")
            else:
                print(f"{indent}{prefix}: Split on {node.split_feature} <= {node.split_value} "
                      f"(samples={node.n_samples}, fraud_rate={node.n_fraud/node.n_samples:.3f})")
                
                if depth < max_depth:
                    print_node(node.left_child, depth + 1, "├─ Left")
                    print_node(node.right_child, depth + 1, "└─ Right")
        
        print("\n Tree Structure:")
        print("="*60)
        print_node(self.root)

# Function to evaluate the decision tree on synthetic fraud data by testing different split criteria, chi-square thresholds, and reporting performance metrics.
def test_decision_tree():
    
    print("\n" + "="*70)
    print("TESTING DECISION TREE IMPLEMENTATION")
    print("="*70)
    
    # Create synthetic fraud detection data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Generate features
    X = pd.DataFrame({
        'transaction_amount': np.random.exponential(50, n_samples),
        'days_since_last': np.random.randint(0, 30, n_samples),
        'num_transactions': np.random.poisson(3, n_samples),
        'merchant_risk': np.random.uniform(0, 1, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples)
    })
    
    # Generate labels
    fraud_probability = (
        (X['transaction_amount'] > 200) * 0.3 +
        (X['days_since_last'] < 1) * 0.2 +
        (X['num_transactions'] > 5) * 0.2 +
        (X['merchant_risk'] > 0.7) * 0.3
    )
    y = (np.random.random(n_samples) < fraud_probability).astype(int)
    
    print(f"Dataset created: {n_samples} samples, {np.sum(y)} fraud cases ({np.mean(y)*100:.1f}%)")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Test different criteria
    print("\n" + "="*60)
    print("Testing Different Split Criteria")
    print("="*60)
    
    for criterion in ['entropy', 'gini', 'misclassification']:
        print(f"\n{criterion.upper()} Criterion:")
        print("-"*40)
        
        tree = DecisionTree(
            criterion=criterion,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            chi_square_alpha=0.05
        )
        
        tree.fit(X_train, y_train)
        
        # Calculate metrics
        train_acc = tree.calculate_accuracy(X_train, y_train)
        test_acc = tree.calculate_accuracy(X_test, y_test)
        train_balanced = tree.calculate_balanced_accuracy(X_train, y_train)
        test_balanced = tree.calculate_balanced_accuracy(X_test, y_test)
        
        # Get tree statistics
        stats = tree.get_tree_stats()
        
        print(f"Tree Statistics:")
        print(f"  Nodes: {stats['total_nodes']}, Leaves: {stats['total_leaves']}")
        print(f"  Max depth: {stats['max_depth']}, Avg depth: {stats['avg_depth']:.2f}")
        
        print(f"Performance:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Train Balanced Accuracy: {train_balanced:.4f}")
        print(f"  Test Balanced Accuracy: {test_balanced:.4f}")
    
    # Test different alpha values for chi-square
    print("\n" + "="*60)
    print("Testing Different Chi-Square Alpha Values")
    print("="*60)
    
    alpha_values = [0.01, 0.05, 0.1, 0.25, 0.5]
    
    for alpha in alpha_values:
        tree = DecisionTree(
            criterion='entropy',
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            chi_square_alpha=alpha
        )
        
        tree.fit(X_train, y_train)
        stats = tree.get_tree_stats()
        test_balanced = tree.calculate_balanced_accuracy(X_test, y_test)
        
        print(f"\n Alpha={alpha}: Nodes={stats['total_nodes']}, "
              f"Depth={stats['max_depth']}, "
              f"Test Balanced Acc={test_balanced:.4f}")
    
    # Print sample tree structure
    print("\n" + "="*60)
    print("Sample Tree Structure (Alpha=0.05)")
    print("="*60)
    
    tree = DecisionTree(
        criterion='entropy',
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        chi_square_alpha=0.05
    )
    tree.fit(X_train, y_train)
    tree.print_tree_structure(max_depth=3)
    
    print("\n Decision Tree implementation complete and tested!")

if __name__ == "__main__":
    test_decision_tree()
