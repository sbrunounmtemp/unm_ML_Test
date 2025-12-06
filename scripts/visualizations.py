#!/bin/python

# This script generates plots for our Random Forest project, including split criteria comparisons, chi-square effects, and feature importance.

import matplotlib.pyplot as plt
import numpy as np

# Set style for better-looking graphs
plt.style.use('seaborn-v0_8-darkgrid')

# Function to create bar chart comparing different split criteria
def create_criteria_comparison():
    
    criteria = ['Entropy', 'Gini', 'Misclassification']
    balanced_acc = [0.5401, 0.5600, 0.5000]
    nodes = [21, 19, 1]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Balanced Accuracy comparison
    bars1 = ax1.bar(criteria, balanced_acc, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax1.set_ylabel('Balanced Accuracy', fontsize=12)
    ax1.set_title('Performance by Split Criterion', fontsize=14, fontweight='bold')
    ax1.set_ylim(0.45, 0.60)
    
    # Add value labels on bars
    for bar, val in zip(bars1, balanced_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{val:.4f}', ha='center', fontsize=10)
    
    # Tree complexity comparison
    bars2 = ax2.bar(criteria, nodes, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax2.set_ylabel('Number of Nodes', fontsize=12)
    ax2.set_title('Tree Complexity by Split Criterion', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, val in zip(bars2, nodes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('criteria_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: criteria_comparison.png")

# Function to create line plot showing chi-square alpha effects
def create_alpha_analysis():
    
    alphas = [0.01, 0.05, 0.10, 0.25, 0.50, 0.90]
    nodes = [13, 23, 35, 45, 47, 47]
    depths = [4, 6, 9, 9, 10, 10]
    balanced_acc = [0.5031, 0.5401, 0.5317, 0.5802, 0.5802, 0.5802]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Tree complexity vs alpha
    ax1.plot(alphas, nodes, marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Nodes')
    ax1.plot(alphas, [d*4 for d in depths], marker='s', linewidth=2, markersize=8, color='#A23B72', label='Depth×4')
    ax1.set_xlabel('Chi-square Alpha (α)', fontsize=12)
    ax1.set_ylabel('Tree Complexity', fontsize=12)
    ax1.set_title('Tree Structure vs Alpha Value', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Balanced accuracy vs alpha
    ax2.plot(alphas, balanced_acc, marker='o', linewidth=2, markersize=8, color='#F18F01')
    ax2.set_xlabel('Chi-square Alpha (α)', fontsize=12)
    ax2.set_ylabel('Balanced Accuracy', fontsize=12)
    ax2.set_title('Performance vs Alpha Value', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.48, 0.60)
    ax2.grid(True, alpha=0.3)
    
    # Mark optimal alpha
    optimal_idx = balanced_acc.index(max(balanced_acc))
    ax2.scatter(alphas[optimal_idx], balanced_acc[optimal_idx], s=200, color='red', 
               zorder=5, alpha=0.5, label=f'Optimal α={alphas[optimal_idx]}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('alpha_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: alpha_analysis.png")

# Function to create horizontal bar chart of feature importance
def create_feature_importance():
    
    # Top 10 features from our analysis
    features = ['C6', 'C5', 'C1', 'addr1', 'C9', 'C7', 'ProductCD', 'card1', 'C14', 'C13']
    importance = [0.4000, 0.4000, 0.4000, 0.3000, 0.3000, 0.2500, 0.2000, 0.2000, 0.1500, 0.1500]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importance, color='#2E86AB')
    
    # Customize chart
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Feature Importance (Usage Rate)', fontsize=12)
    ax.set_title('Top 10 Most Important Features for Fraud Detection', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{val:.1%}', va='center', fontsize=10)
    
    ax.set_xlim(0, 0.45)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: feature_importance.png")

# Function to create a summary visualization of model performance
def create_performance_summary():
    
    models = ['Single Tree\n(Baseline)', 'Our RF\n(10 trees)', 'Sklearn RF\n(20 trees)']
    balanced_acc = [0.5451, 0.5555, 0.6081]
    training_time = [0.5, 158.65, 0.72]  
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    bars1 = ax1.bar(models, balanced_acc, color=['#F18F01', '#A23B72', '#2E86AB'])
    ax1.set_ylabel('Balanced Accuracy', fontsize=12)
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0.50, 0.65)
    
    # Add baseline line
    ax1.axhline(y=0.50, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    ax1.legend()
    
    for bar, val in zip(bars1, balanced_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
                f'{val:.4f}', ha='center', fontsize=10)
    
    # Training time comparison (log scale)
    bars2 = ax2.bar(models, training_time, color=['#F18F01', '#A23B72', '#2E86AB'])
    ax2.set_ylabel('Training Time (seconds, log scale)', fontsize=12)
    ax2.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    
    for bar, val in zip(bars2, training_time):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5, 
                f'{val:.1f}s', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: performance_summary.png")

# Function to generate all visualizations
def main():

    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    print("\n1. Creating split criteria comparison...")
    create_criteria_comparison()
    
    print("\n2. Creating chi-square alpha analysis...")
    create_alpha_analysis()
    
    print("\n3. Creating feature importance chart...")
    create_feature_importance()
    
    print("\n4. Creating performance summary...")
    create_performance_summary()
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print("\n Generated files:")
    print("  - criteria_comparison.png")
    print("  - alpha_analysis.png")
    print("  - feature_importance.png")
    print("  - performance_summary.png")

if __name__ == "__main__":
    main()
