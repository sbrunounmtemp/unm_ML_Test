#!/bin/python

# This script implements a Chi-Square test to check if decision tree splits are statistically significant to help avoid overfitting.

import numpy as np
from scipy.stats import chi2

# Class for applying the Chi-Square test to decide if a decision tree split is statistically significant.
class ChiSquareTest:
    
    # Function to initialize Chi-Square test calculation with a minimum sample size requirement for validity.
    def __init__(self):
        self.min_samples_for_test = 5
        
    # Function to compute chi-square statistic and p-value to test if a split yields a significant class distribution difference from its parent.
    def calculate_chi_square_statistic(self, y_parent, y_left, y_right):
        
        # Get unique classes
        classes = np.unique(y_parent)
        n_classes = len(classes)
        
        # Build a contingency table of observed class frequencies for left and right child nodes.
        observed_frequencies = np.zeros((2, n_classes))
        
        for i, class_label in enumerate(classes):
            observed_frequencies[0, i] = np.sum(y_left == class_label)  # left child
            observed_frequencies[1, i] = np.sum(y_right == class_label)  # right child
        
        # Calculate expected frequencies under null hypothesis
        total_left = len(y_left)
        total_right = len(y_right)
        total_samples = total_left + total_right
        
        expected_frequencies = np.zeros((2, n_classes))
        
        for i, class_label in enumerate(classes):
            # Proportion of this class in parent
            class_proportion = np.sum(y_parent == class_label) / len(y_parent)
            
            # Expected count in each child under null hypothesis
            expected_frequencies[0, i] = total_left * class_proportion
            expected_frequencies[1, i] = total_right * class_proportion
        
        # Calculate chi-square statistic
        chi_square_value = 0
        
        for i in range(2):  # For each child
            for j in range(n_classes):  # For each class
                expected = expected_frequencies[i, j]
                if expected > 0:  # Avoid division by zero
                    observed = observed_frequencies[i, j]
                    chi_square_value += ((observed - expected) ** 2) / expected
        
        # Calculate the degrees of freedom for the chi-square test
        degrees_of_freedom = (2 - 1) * (n_classes - 1)
        
        # Calculate p-value using chi-square distribution
        p_value = 1 - chi2.cdf(chi_square_value, degrees_of_freedom)
        return chi_square_value, p_value, degrees_of_freedom
    
    
    # Function to apply chi-square test to decide whether a decision tree split is statistically significant or should be stopped.
    def should_stop_splitting(self, y_parent, y_left, y_right, alpha=0.05):
        
        # Check if we have enough samples for a valid test
        if (len(y_left) < self.min_samples_for_test or 
            len(y_right) < self.min_samples_for_test):
            return True, {
                'reason': 'Too few samples for chi-square test',
                'left_samples': len(y_left),
                'right_samples': len(y_right),
                'min_required': self.min_samples_for_test
            }
        
        # Calculate chi-square statistic
        chi_sq_value, p_value, df = self.calculate_chi_square_statistic(
            y_parent, y_left, y_right
        )
        
        # Critical value from chi-square distribution
        critical_value = chi2.ppf(1 - alpha, df)
        
        # Decision to stop if p-value > alpha (split not significant)
        should_stop = p_value > alpha
        
        test_info = {
            'chi_square_statistic': chi_sq_value,
            'p_value': p_value,
            'alpha': alpha,
            'critical_value': critical_value,
            'degrees_of_freedom': df,
            'should_stop': should_stop,
            'decision': 'STOP (not significant)' if should_stop else 'CONTINUE (significant)',
            'left_samples': len(y_left),
            'right_samples': len(y_right),
            'left_fraud_rate': np.mean(y_left) if len(y_left) > 0 else 0,
            'right_fraud_rate': np.mean(y_right) if len(y_right) > 0 else 0
        }
        
        return should_stop, test_info
    
    # Function to demonstrate chi-square test through example splits to show when a decision tree should stop or continue splitting.
    def demonstrate_chi_square_test(self):
       
        print("="*70)
        print("CHI-SQUARE TEST DEMONSTRATION")
        print("="*70)
        
        # Perfect split (highly significant)
        print("\n1. PERFECT SPLIT (should be significant)")
        parent_1 = np.array([0]*50 + [1]*50)
        left_1 = np.array([0]*50)  # All normal
        right_1 = np.array([1]*50)  # All fraud
        
        should_stop_1, info_1 = self.should_stop_splitting(parent_1, left_1, right_1, alpha=0.05)
        self._print_test_results(info_1)
        
        # Random split (not significant)
        print("\n2. RANDOM SPLIT (should not be significant)")
        np.random.seed(42)
        parent_2 = np.array([0]*96 + [1]*4)  # 4% fraud rate
        indices = np.random.permutation(100)
        left_2 = parent_2[indices[:50]]
        right_2 = parent_2[indices[50:]]
        
        should_stop_2, info_2 = self.should_stop_splitting(parent_2, left_2, right_2, alpha=0.05)
        self._print_test_results(info_2)
        
        # Good but not perfect split (likely significant)
        print("\n3. GOOD SPLIT (should be significant)")
        parent_3 = np.array([0]*965 + [1]*35)  # 3.5% fraud like our data

        # Put most fraud in left node
        left_3 = np.array([0]*20 + [1]*30)  # 60% fraud rate
        right_3 = np.array([0]*945 + [1]*5)  # 0.5% fraud rate
        
        should_stop_3, info_3 = self.should_stop_splitting(parent_3, left_3, right_3, alpha=0.05)
        self._print_test_results(info_3)
        
        # Test different alpha values
        print("\n4. TESTING DIFFERENT ALPHA VALUES")
        print("   Using the good split.")
        
        alpha_values = [0.01, 0.05, 0.1, 0.25, 0.5, 0.9]
        for alpha in alpha_values:
            should_stop, info = self.should_stop_splitting(parent_3, left_3, right_3, alpha)
            print(f"\n   Alpha = {alpha:.2f}:")
            print(f"     p-value = {info['p_value']:.6f}")
            print(f"     Decision: {info['decision']}")
            print(f"     Critical value: {info['critical_value']:.4f}")
            print(f"     Chi-square stat: {info['chi_square_statistic']:.4f}")
    
    
    # Function to display chai-square test results.
    def _print_test_results(self, info):
        
        print(f"   Left: {info['left_samples']} samples, "
              f"{info['left_fraud_rate']*100:.1f}% fraud")
        print(f"   Right: {info['right_samples']} samples, "
              f"{info['right_fraud_rate']*100:.1f}% fraud")
        print(f"   Chi-square statistic: {info['chi_square_statistic']:.4f}")
        print(f"   p-value: {info['p_value']:.6f}")
        print(f"   Alpha: {info['alpha']}")
        print(f"   Decision: {info['decision']}")
    
    
    # Function to simulate fraud detection with class imbalance to test chi-square stopping decisions under realistic conditions.
    def test_with_fraud_data_simulation(self):
        
        print("\n" + "="*70)
        print("TESTING WITH FRAUD-LIKE DATA (3.5% fraud rate)")
        print("="*70)
        
        np.random.seed(42)
        
        # Create parent node with fraud rate similar to our data
        n_samples = 10000
        n_fraud = int(n_samples * 0.035)
        n_normal = n_samples - n_fraud
        
        parent = np.array([0] * n_normal + [1] * n_fraud)
        np.random.shuffle(parent)
        
        print(f"\n Parent node: {n_samples} samples, {n_fraud} fraud ({n_fraud/n_samples*100:.1f}%)")
        
        # Test 1: Feature that catches fraud well
        print("\n1. GOOD FEATURE (catches 70% of fraud in 5% of data):")
        fraud_indices = np.where(parent == 1)[0]
        normal_indices = np.where(parent == 0)[0]
        
        # Select indices for left node
        left_fraud = np.random.choice(fraud_indices, size=int(n_fraud * 0.7), replace=False)
        left_normal = np.random.choice(normal_indices, size=int(n_normal * 0.03), replace=False)
        left_indices = np.concatenate([left_fraud, left_normal])
        
        # Allocate remaining indices to right node 
        right_indices = np.setdiff1d(np.arange(n_samples), left_indices)
        
        left = parent[left_indices]
        right = parent[right_indices]
        
        should_stop, info = self.should_stop_splitting(parent, left, right, alpha=0.05)
        self._print_test_results(info)
        
        # Poor feature (random split)
        print("\n2. POOR FEATURE (random split):")
        shuffled = np.random.permutation(n_samples)
        split_point = n_samples // 2
        
        left_random = parent[shuffled[:split_point]]
        right_random = parent[shuffled[split_point:]]
        
        should_stop_random, info_random = self.should_stop_splitting(
            parent, left_random, right_random, alpha=0.05
        )
        self._print_test_results(info_random)

# Main execution block to execute the chi-square module by running illustrative examples and fraud-simulation tests to validate its role in decision tree stopping.
if __name__ == "__main__":
    print("CHI-SQUARE TEST IMPLEMENTATION")
    print("="*70)
    
    # Create instance
    chi_square_tester = ChiSquareTest()
    
    # Run demonstrations
    chi_square_tester.demonstrate_chi_square_test()
    
    # Test with fraud-like data
    chi_square_tester.test_with_fraud_data_simulation()
    
    print("\n Chi-Square test implemented to prevent overfitting and stop splits that are not statistically significantl")
   
