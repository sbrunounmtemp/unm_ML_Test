#!/bin/python

# This script performs the initial setup for our Random Forest project, including dataset loading and preliminary preparation for analysis.

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
# Removed plotting and chi2 imports from this setup module (not used here)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# Class to handle all data loading and preprocessing operations.
class DataLoader:

    # Function to set the dataset path, initializes data placeholders, and specifies categorical features for the project.
    def __init__(self, data_folder_path, nrows=None, use_parquet=True):

        self.data_folder_path = data_folder_path
        self.train_data = None
        self.test_data = None
        self.sample_submission = None
        self.nrows = nrows
        self.use_parquet = use_parquet
        self.parquet_train = os.path.join(self.data_folder_path, 'train_clean.parquet')
        self.parquet_test = os.path.join(self.data_folder_path, 'test_clean.parquet')
        self.categorical_features = [
            'ProductCD', 
            'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2'
        ]

    # Function to Load the training, testing, and sample submission datasets into memory for analysis.    
    def load_data(self):
       
        try:
            # Load training data
            train_path = os.path.join(self.data_folder_path, 'train.csv')
            self.train_data = pd.read_csv(train_path)
            print(f"Training data loaded successfully: {self.train_data.shape}")
            
            # Load test data
            test_path = os.path.join(self.data_folder_path, 'test.csv')
            self.test_data = pd.read_csv(test_path)
            print(f"Test data loaded successfully: {self.test_data.shape}")
            
            # Load sample submission
            sample_sub_path = os.path.join(self.data_folder_path, 'sample_sub.csv')
            self.sample_submission = pd.read_csv(sample_sub_path)
            print(f"Sample submission loaded successfully: {self.sample_submission.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    # Function to perform preliminary exploration of the training dataset to check its structure and readiness for analysis.
    def explore_data(self):
        
        if self.train_data is None:
            print("Load data using load_data()")
            return
        
        print("\n" + "="*50)
        print("DATA EXPLORATION SUMMARY")
        print("="*50)
        
        # Basic information
        print("\n1. Dataset Shape:")
        print(f"   Training samples: {len(self.train_data)}")
        print(f"   Test samples: {len(self.test_data)}")
        print(f"   Number of features: {len(self.train_data.columns) - 2}")  
        
        # Target distribution
        print("\n2. Target Variable Distribution (isFraud):")
        fraud_counts = self.train_data['isFraud'].value_counts()
        print(f"   Non-Fraud (0): {fraud_counts[0]} ({fraud_counts[0]/len(self.train_data)*100:.2f}%)")
        print(f"   Fraud (1): {fraud_counts[1]} ({fraud_counts[1]/len(self.train_data)*100:.2f}%)")
        print(f"   Class Imbalance Ratio: 1:{fraud_counts[0]/fraud_counts[1]:.1f}")
        
        # Missing values
        print("\n3. Missing Values in Training Data:")
        missing_counts = self.train_data.isnull().sum()
        missing_features = missing_counts[missing_counts > 0]
        if len(missing_features) > 0:
            print(f"   Features with missing values: {len(missing_features)}/{len(self.train_data.columns)}")
            print("   Top 5 features with most missing values:")
            for feature, count in missing_features.nlargest(5).items():
                print(f"     - {feature}: {count} ({count/len(self.train_data)*100:.2f}%)")
        else:
            print("   No missing values found")
        
        # Data types
        print("\n4. Data Types:")
        print(f"   Numeric features: {len(self.train_data.select_dtypes(include=[np.number]).columns)}")
        print(f"   String features: {len(self.train_data.select_dtypes(include=['object']).columns)}")
        
        # Feature columns
        print("\n5. Feature Columns:")
        print("   All columns:", list(self.train_data.columns[:10]), "...")
        print("   Categorical features specified:", self.categorical_features)
        
        return self.train_data.head()
    
    # Function to split the dataset into training and validation sets to support our model development and evaluation.
    def prepare_data_for_training(self, validation_split=0.2):
        
        if self.train_data is None:
            print("Load data using load_data()")
            return None
        
        # Separate features and target and remove TransactionID because it is just an identifier
        feature_columns = [col for col in self.train_data.columns 
                          if col not in ['TransactionID', 'isFraud']]
        
        X = self.train_data[feature_columns]
        y = self.train_data['isFraud']
        
        # Split the data and maintain class distribution 
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=validation_split, 
            random_state=RANDOM_SEED,
            stratify=y  
        )
        
        print(f"\n Data Split Complete:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Training fraud rate: {y_train.mean()*100:.2f}%")
        print(f"  Validation fraud rate: {y_val.mean()*100:.2f}%")
        
        return X_train, X_val, y_train, y_val

# Main execution block that executes the setup process by loading, exploring, and preparing the dataset for our Random Forest training.
if __name__ == "__main__":
    
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_path)  
    DATA_FOLDER = os.path.join(project_root, "data")
    
    print("="*50)
    print("DATA LOADING")
    print(f"Looking for data in: {DATA_FOLDER}")
    
    # Initialize data loader
    loader = DataLoader(DATA_FOLDER)
    
    # Load the data and explore the data
    if loader.load_data():
        data_head = loader.explore_data()
        print("\nFirst 5 rows of training data:")
        print(data_head)
        
        # Prepare data for training
        X_train, X_val, y_train, y_val = loader.prepare_data_for_training()
        
        print("\n Data loaded and ready for processing!")
    else:
        print("\n Failed to load data.")
