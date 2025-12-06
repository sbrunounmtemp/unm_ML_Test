#!/bin/python

# This script implements entropy, gini index, and misclassification error to compute information gain for decision tree splits.

import numpy as np
import pandas as pd

# Class containing all information gain measures for decision tree splitting.
class InformationGainMeasures:
    
    # Function to initialize infomation gain measures to ensure stable information gain computations.
    def __init__(self):
        self.epsilon = 1e-10  
    

    # ENTROPY IMPLEMENTATION

    # Function to compute entropy of class labels to measure impurity and randomness in the dataset split.
    def calculate_entropy(self, y_values):
        
        # Entropy is 0 if array is empty
        if len(y_values) == 0:
            return 0
        
        # Count occurrences of each class
        total_samples = len(y_values)
        unique_classes, class_counts = np.unique(y_values, return_counts=True)
        
        # Calculate entropy
        entropy = 0
        for count in class_counts:
            if count > 0:  
                probability = count / total_samples
                entropy -= probability * np.log2(probability + self.epsilon)
        
        return entropy
    

    # GINI INDEX IMPLEMENTATION
    
    # Function to compute gini index of class labels to quantify node impurity in decision tree splitting.
    def calculate_gini_index(self, y_values):
       
        # Gini index is 0 if array is empty
        if len(y_values) == 0:
            return 0
        
        # Count occurrences of each class
        total_samples = len(y_values)
        unique_classes, class_counts = np.unique(y_values, return_counts=True)
        
        # Calculate gini index
        gini = 1.0
        for count in class_counts:
            probability = count / total_samples
            gini -= probability ** 2
        
        return gini
    
    
    # MISCLASSIFICATION ERROR IMPLEMENTATION
    
    # Function to compute misclassification error to estimate the proportion of incorrectly classified samples in a node.
    def calculate_misclassification_error(self, y_values):
        
        # Misclassification error is 0 if array is empty
        if len(y_values) == 0:
            return 0
        
        # Count occurrences of each class
        total_samples = len(y_values)
        unique_classes, class_counts = np.unique(y_values, return_counts=True)
        
        # Calculate misclassification error
        max_probability = np.max(class_counts) / total_samples
        misclassification_error = 1 - max_probability
        
        return misclassification_error
    
    
    # INFORMATION GAIN CALCULATION
    
    # Function to compute information gain of a split to determine how well it reduces impurity in decision tree learning.
    def calculate_information_gain(self, parent_y, left_child_y, right_child_y, criterion='entropy'):
        
        # Select the appropriate impurity measure
        if criterion == 'entropy':
            impurity_function = self.calculate_entropy
        elif criterion == 'gini':
            impurity_function = self.calculate_gini_index
        elif criterion == 'misclassification':
            impurity_function = self.calculate_misclassification_error
        else:
            raise ValueError(f"Unknown criterion: {criterion}. Use 'entropy', 'gini', or 'misclassification'")
        
        # Calculate parent impurity
        parent_impurity = impurity_function(parent_y)
        
        # Calculate weighted average of child impurities
        total_samples = len(parent_y)
        left_weight = len(left_child_y) / total_samples
        right_weight = len(right_child_y) / total_samples
        
        left_impurity = impurity_function(left_child_y)
        right_impurity = impurity_function(right_child_y)
        
        weighted_children_impurity = (left_weight * left_impurity + 
                                     right_weight * right_impurity)
        
        # Information gain is the reduction in impurity
        information_gain = parent_impurity - weighted_children_impurity
        
        return information_gain
    
    
    # DEMONSTRATION AND TESTING
    
    # Function to demonstrate entropy, Gini index, misclassification error, and information gain.
    def demonstrate_measures(self):
       
        print("="*60)
        print("DEMONSTRATION OF INFORMATION GAIN MEASURES")
        print("="*60)
        
        # Pure node (all same class)
        pure_node = np.array([0, 0, 0, 0, 0])
        print("\n1. PURE NODE (all class 0):", pure_node)
        print(f"   Entropy: {self.calculate_entropy(pure_node):.4f}")
        print(f"   Gini Index: {self.calculate_gini_index(pure_node):.4f}")
        print(f"   Misclassification Error: {self.calculate_misclassification_error(pure_node):.4f}")
        
        # Perfectly mixed node (50-50 split)
        mixed_node = np.array([0, 1, 0, 1, 0, 1])
        print("\n2. PERFECTLY MIXED NODE (50-50):", mixed_node)
        print(f"   Entropy: {self.calculate_entropy(mixed_node):.4f}")
        print(f"   Gini Index: {self.calculate_gini_index(mixed_node):.4f}")
        print(f"   Misclassification Error: {self.calculate_misclassification_error(mixed_node):.4f}")
        
        # Imbalanced node (like our fraud data)
        imbalanced_node = np.array([0]*96 + [1]*4)  # 4% fraud rate
        print("\n3. IMBALANCED NODE (4% class 1, 96% class 0):")
        print(f"   Entropy: {self.calculate_entropy(imbalanced_node):.4f}")
        print(f"   Gini Index: {self.calculate_gini_index(imbalanced_node):.4f}")
        print(f"   Misclassification Error: {self.calculate_misclassification_error(imbalanced_node):.4f}")
        
        # Information Gain calculation
        print("\n4. INFORMATION GAIN EXAMPLE:")
        parent = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
        left_child = np.array([0, 0, 0, 0, 0])  # Pure split
        right_child = np.array([1, 1, 1, 1, 1])  # Pure split
        
        print(f"   Parent: {parent}")
        print(f"   Left child: {left_child}")
        print(f"   Right child: {right_child}")
        
        for criterion in ['entropy', 'gini', 'misclassification']:
            ig = self.calculate_information_gain(parent, left_child, right_child, criterion)
            print(f"   Information Gain ({criterion}): {ig:.4f}")

    
    # Function to simulate fraud detection on the dataset to test impurity measures and information gain under realistic class imbalance.
    def test_with_real_data_simulation(self):
        
        print("\n" + "="*60)
        print("TESTING WITH FRAUD-LIKE DATA (3.5% fraud rate)")
        print("="*60)
        
        # Simulate our actual data distribution
        np.random.seed(42)
        n_samples = 1000
        fraud_rate = 0.035  
        
        # Create imbalanced dataset
        n_fraud = int(n_samples * fraud_rate)
        n_normal = n_samples - n_fraud
        
        y_data = np.array([0] * n_normal + [1] * n_fraud)
        np.random.shuffle(y_data)
        
        print(f"\n Dataset: {n_samples} samples, {n_fraud} fraud, {n_normal} normal")
        print(f"Fraud rate: {fraud_rate*100:.1f}%")
        
        # Calculate impurity measures
        print(f"\n Impurity Measures:")
        print(f"  Entropy: {self.calculate_entropy(y_data):.4f}")
        print(f"  Gini Index: {self.calculate_gini_index(y_data):.4f}")
        print(f"  Misclassification Error: {self.calculate_misclassification_error(y_data):.4f}")
        
        # Simulate a good split (separates fraud cases well)
        good_split_indices = np.random.choice(np.where(y_data == 1)[0], 
                                            size=int(n_fraud * 0.8), 
                                            replace=False)
        normal_in_split = int(n_normal * 0.05)  # 5% false positives
        normal_indices = np.random.choice(np.where(y_data == 0)[0], 
                                         size=normal_in_split, 
                                         replace=False)
        
        left_indices = np.concatenate([good_split_indices, normal_indices])
        right_indices = np.setdiff1d(np.arange(n_samples), left_indices)
        
        left_y = y_data[left_indices]
        right_y = y_data[right_indices]
        
        print(f"\n Simulated Split:")
        print(f"  Left node: {len(left_y)} samples, {np.sum(left_y)} fraud")
        print(f"  Right node: {len(right_y)} samples, {np.sum(right_y)} fraud")
        
        print(f"\n Information Gain for this split:")
        for criterion in ['entropy', 'gini', 'misclassification']:
            ig = self.calculate_information_gain(y_data, left_y, right_y, criterion)
            print(f"  {criterion}: {ig:.4f}")

# Main execution block to execute information gain module by running illustrative examples and fraud-simulation tests to validate the measures.
if __name__ == "__main__":
    print("INFORMATION GAIN MEASURES IMPLEMENTATION")
    print("="*60)
    
    # Create instance
    ig_calculator = InformationGainMeasures()
    
    # Run demonstrations
    ig_calculator.demonstrate_measures()
    
    # Test with fraud-like data
    ig_calculator.test_with_real_data_simulation()
    
    print("\n Information Gain measures implemented. Intended to guide our feature selection and split evaluation in our Decision Tree.")
    
