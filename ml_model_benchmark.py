"""
Multi-Model Benchmark for HR Attrition Prediction
==================================================
Comprehensive comparison of 5 classification algorithms with proper preprocessing,
SMOTE for class imbalance, and business-focused evaluation metrics.

Models Compared:
1. Random Forest Classifier
2. Decision Tree Classifier
3. Logistic Regression
4. Support Vector Machine (SVM)
5. Perceptron

Author: Lead Data Scientist - HR Analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import sys
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE

# Configuration
warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Create output directories
OUTPUT_DIR = 'outputs/model_results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[OK] Created output directory: {OUTPUT_DIR}/")


def preprocess_data(df):
    """
    COMPREHENSIVE PREPROCESSING PIPELINE
    
    Steps:
    1. Categorical encoding (ordinal + one-hot)
    2. Feature separation (X and y)
    3. Feature scaling (StandardScaler for numeric features)
    4. Train-test split (80/20, stratified)
    5. SMOTE application (training set only)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw master attrition dataset
    
    Returns:
    --------
    tuple: (X_train_res, X_test, y_train_res, y_test, feature_names, scaler, 
            train_distribution, test_distribution)
    """
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE")
    print("="*70)
    
    df_encoded = df.copy()
    
    # 1. CATEGORICAL ENCODING
    print("\n[STEP 1] Categorical Variable Encoding")
    print("-"*70)
    
    # Ordinal variables (with meaningful order)
    if 'BusinessTravel' in df_encoded.columns:
        business_travel_map = {
            'Non-Travel': 0,
            'Travel_Rarely': 1,
            'Travel_Frequently': 2
        }
        df_encoded['BusinessTravel'] = df_encoded['BusinessTravel'].map(business_travel_map)
        print("  [OK] BusinessTravel: Ordinal encoding (0=Non, 1=Rarely, 2=Frequently)")
    
    # Nominal variables (no inherent order - use One-Hot Encoding)
    nominal_columns = []
    for col in ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']:
        if col in df_encoded.columns:
            nominal_columns.append(col)
    
    if nominal_columns:
        df_encoded = pd.get_dummies(df_encoded, columns=nominal_columns, drop_first=True)
        print(f"  [OK] One-Hot Encoding: {len(nominal_columns)} nominal variables")
        print(f"       Variables: {', '.join(nominal_columns)}")
    
    # Drop text version of Attrition if present
    if 'Attrition' in df_encoded.columns:
        df_encoded = df_encoded.drop('Attrition', axis=1)
    
    print(f"  [OK] Dataset shape after encoding: {df_encoded.shape}")
    
    # 2. FEATURE SEPARATION
    print("\n[STEP 2] Feature Separation")
    print("-"*70)
    
    y = df_encoded['Attrition_Binary']
    X = df_encoded.drop(['Attrition_Binary', 'EmployeeID'], axis=1, errors='ignore')
    
    print(f"  [OK] Features (X): {X.shape}")
    print(f"  [OK] Target (y): {y.shape}")
    print(f"  [OK] Class distribution: {y.value_counts().to_dict()}")
    
    feature_names = X.columns.tolist()
    
    # 3. FEATURE SCALING
    print("\n[STEP 3] Feature Scaling")
    print("-"*70)
    
    # Identify numeric columns for scaling (exclude binary one-hot encoded features)
    numeric_features = []
    for col in X.columns:
        # Check if column has more than 2 unique values (not just 0/1)
        if X[col].dtype in ['float64', 'int64'] and X[col].nunique() > 2:
            numeric_features.append(col)
    
    print(f"  [OK] Identified {len(numeric_features)} numeric features for scaling")
    
    # Apply StandardScaler to numeric features
    scaler = StandardScaler()
    X_scaled = X.copy()
    
    if numeric_features:
        X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
        print(f"  [OK] Applied StandardScaler to numeric features")
        print(f"       Critical for: SVM, Logistic Regression, Perceptron")
    
    # 4. TRAIN-TEST SPLIT
    print("\n[STEP 4] Train-Test Split")
    print("-"*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"  [OK] Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  [OK] Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    train_dist_before = y_train.value_counts()
    print(f"\n  Training set class distribution (before SMOTE):")
    print(f"    - Class 0 (Stay): {train_dist_before[0]} ({train_dist_before[0]/len(y_train)*100:.1f}%)")
    print(f"    - Class 1 (Leave): {train_dist_before[1]} ({train_dist_before[1]/len(y_train)*100:.1f}%)")
    
    # 5. SMOTE APPLICATION (Training Set Only)
    print("\n[STEP 5] SMOTE Class Balancing")
    print("-"*70)
    print("  Business Rationale: Generate synthetic minority samples to help")
    print("  the model learn patterns from underrepresented attrition cases.")
    print("  CRITICAL: Applied ONLY to training set to avoid data leakage.")
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    train_dist_after = y_train_res.value_counts()
    print(f"\n  Training set class distribution (after SMOTE):")
    print(f"    - Class 0 (Stay): {train_dist_after[0]} ({train_dist_after[0]/len(y_train_res)*100:.1f}%)")
    print(f"    - Class 1 (Leave): {train_dist_after[1]} ({train_dist_after[1]/len(y_train_res)*100:.1f}%)")
    print(f"  [OK] Created {train_dist_after[1] - train_dist_before[1]} synthetic samples")
    
    return (X_train_res, X_test, y_train_res, y_test, feature_names, scaler,
            train_dist_before, train_dist_after)


def initialize_models():
    """
    Initialize 5 classification models with optimal configurations.
    
    Returns:
    --------
    dict: Dictionary with model objects and metadata
    """
    print("\n" + "="*70)
    print("MODEL INITIALIZATION")
    print("="*70)
    
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=None,
                n_jobs=-1
            ),
            'description': 'Ensemble of 100 decision trees',
            'best_for': 'Non-linear patterns, feature importance',
            'color': '#2ecc71'
        },
        
        'Decision Tree': {
            'model': DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
                min_samples_split=20
            ),
            'description': 'Single decision tree with pruning',
            'best_for': 'Interpretability, simple rules',
            'color': '#3498db'
        },
        
        'Logistic Regression': {
            'model': LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='lbfgs'
            ),
            'description': 'Linear probabilistic classifier',
            'best_for': 'Linear relationships, speed',
            'color': '#9b59b6'
        },
        
        'SVM': {
            'model': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                gamma='scale'
            ),
            'description': 'Support Vector Machine with RBF kernel',
            'best_for': 'Complex decision boundaries',
            'color': '#e74c3c'
        },
        
        'Perceptron': {
            'model': Perceptron(
                random_state=42,
                max_iter=1000,
                tol=1e-3
            ),
            'description': 'Single-layer neural network',
            'best_for': 'Simple linear separation',
            'color': '#f39c12'
        }
    }
    
    print(f"\n[OK] Initialized {len(models)} classification models:\n")
    for name, config in models.items():
        print(f"  - {name:20s}: {config['description']}")
        print(f"    {'':20s}  Best for: {config['best_for']}")
    
    return models


def benchmark_models(models, X_train_res, X_test, y_train_res, y_test):
    """
    Train and evaluate all models with comprehensive metrics.
    INCLUDES OVERFITTING DETECTION: Calculate metrics on both train and test sets.
    
    Parameters:
    -----------
    models : dict
        Dictionary of model configurations
    X_train_res : array-like
        SMOTE-resampled training features
    X_test : array-like
        Test features
    y_train_res : array-like
        SMOTE-resampled training target
    y_test : array-like
        Test target
    
    Returns:
    --------
    dict: Results for all models (includes train and test metrics + overfitting gap)
    """
    print("\n" + "="*70)
    print("MODEL TRAINING & EVALUATION (WITH OVERFITTING DETECTION)")
    print("="*70)
    
    results = {}
    
    for i, (model_name, config) in enumerate(models.items(), 1):
        print(f"\n[{i}/{len(models)}] {model_name}")
        print("-"*70)
        
        model = config['model']
        
        # Train model
        print(f"  [TRAINING] Fitting {model_name}...")
        start_time = time.time()
        model.fit(X_train_res, y_train_res)
        training_time = time.time() - start_time
        print(f"  [OK] Training completed in {training_time:.2f} seconds")
        
        # ==================== TEST SET PREDICTIONS ====================
        y_test_pred = model.predict(X_test)
        
        # Get probability estimates (needed for ROC curve)
        if hasattr(model, 'predict_proba'):
            y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_test_pred_proba = model.decision_function(X_test)
        else:
            y_test_pred_proba = y_test_pred
        
        # Calculate TEST metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        test_conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        # ==================== TRAIN SET PREDICTIONS (for overfitting detection) ====================
        y_train_pred = model.predict(X_train_res)
        
        if hasattr(model, 'predict_proba'):
            y_train_pred_proba = model.predict_proba(X_train_res)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_train_pred_proba = model.decision_function(X_train_res)
        else:
            y_train_pred_proba = y_train_pred
        
        # Calculate TRAIN metrics
        train_accuracy = accuracy_score(y_train_res, y_train_pred)
        train_precision = precision_score(y_train_res, y_train_pred, zero_division=0)
        train_recall = recall_score(y_train_res, y_train_pred)
        train_f1 = f1_score(y_train_res, y_train_pred)
        train_roc_auc = roc_auc_score(y_train_res, y_train_pred_proba)
        
        # ==================== OVERFITTING GAP CALCULATION ====================
        # Positive gap = overfitting (train better than test)
        # Negative gap = underfitting (test better than train - rare but possible with regularization)
        gap_accuracy = train_accuracy - test_accuracy
        gap_precision = train_precision - test_precision
        gap_recall = train_recall - test_recall
        gap_f1 = train_f1 - test_f1
        gap_roc_auc = train_roc_auc - test_roc_auc
        
        # Overall overfitting score (average of gaps)
        overfitting_score = (gap_accuracy + gap_recall + gap_f1 + gap_roc_auc) / 4
        
        # Classify overfitting level
        if overfitting_score < 0.02:
            overfitting_level = "Excellent (No overfitting)"
        elif overfitting_score < 0.05:
            overfitting_level = "Good (Minimal overfitting)"
        elif overfitting_score < 0.10:
            overfitting_level = "Moderate (Some overfitting)"
        elif overfitting_score < 0.20:
            overfitting_level = "High (Significant overfitting)"
        else:
            overfitting_level = "Severe (Extreme overfitting)"
        
        # Store results
        results[model_name] = {
            # Test set metrics (what matters for production)
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1_score': test_f1,
            'test_roc_auc': test_roc_auc,
            'confusion_matrix': test_conf_matrix,
            
            # Train set metrics (for overfitting detection)
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1_score': train_f1,
            'train_roc_auc': train_roc_auc,
            
            # Overfitting gaps
            'gap_accuracy': gap_accuracy,
            'gap_precision': gap_precision,
            'gap_recall': gap_recall,
            'gap_f1': gap_f1,
            'gap_roc_auc': gap_roc_auc,
            'overfitting_score': overfitting_score,
            'overfitting_level': overfitting_level,
            
            # Other info
            'training_time': training_time,
            'y_pred': y_test_pred,
            'y_pred_proba': y_test_pred_proba,
            'model_object': model,
            'color': config['color'],
            
            # Backward compatibility (use test metrics as default)
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'roc_auc': test_roc_auc
        }
        
        # Print TEST metrics
        print(f"\n  [TEST SET METRICS]")
        print(f"    Accuracy:  {test_accuracy:.4f}")
        print(f"    Precision: {test_precision:.4f}")
        print(f"    Recall:    {test_recall:.4f} <- MOST IMPORTANT (catch leavers)")
        print(f"    F1-Score:  {test_f1:.4f}")
        print(f"    ROC-AUC:   {test_roc_auc:.4f}")
        
        # Print TRAIN metrics
        print(f"\n  [TRAIN SET METRICS]")
        print(f"    Accuracy:  {train_accuracy:.4f}")
        print(f"    Recall:    {train_recall:.4f}")
        print(f"    F1-Score:  {train_f1:.4f}")
        print(f"    ROC-AUC:   {train_roc_auc:.4f}")
        
        # Print OVERFITTING ANALYSIS
        print(f"\n  [OVERFITTING ANALYSIS]")
        print(f"    Accuracy Gap:  {gap_accuracy:+.4f}")
        print(f"    Recall Gap:    {gap_recall:+.4f}")
        print(f"    F1 Gap:        {gap_f1:+.4f}")
        print(f"    ROC-AUC Gap:   {gap_roc_auc:+.4f}")
        print(f"    Overall Score: {overfitting_score:.4f}")
        print(f"    Level:         {overfitting_level}")
        
        # Confusion Matrix breakdown
        tn, fp, fn, tp = test_conf_matrix.ravel()
        print(f"\n  [CONFUSION MATRIX - TEST SET]")
        print(f"    True Negatives (TN):  {tn} (Correctly predicted stayers)")
        print(f"    False Positives (FP): {fp} (False alarms)")
        print(f"    False Negatives (FN): {fn} (MISSED LEAVERS - CRITICAL)")
        print(f"    True Positives (TP):  {tp} (Correctly predicted leavers)")
    
    print(f"\n[OK] All {len(models)} models trained and evaluated")
    
    return results


def generate_comparison_visualizations(results, feature_names, y_test, output_dir):
    """
    Generate comprehensive comparison visualizations.
    
    Parameters:
    -----------
    results : dict
        Results from all models
    feature_names : list
        List of feature names
    y_test : array-like
        Test target values
    output_dir : str
        Directory to save outputs
    """
    print("\n" + "="*70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*70)
    
    # 1. Performance Comparison Table (CSV) - WITH OVERFITTING METRICS
    print("\n[VIZ 1/6] Performance Comparison Table (with Overfitting Metrics)")
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Test_Accuracy': f"{metrics['test_accuracy']:.4f}",
            'Test_Recall': f"{metrics['test_recall']:.4f}",
            'Test_F1': f"{metrics['test_f1_score']:.4f}",
            'Test_ROC-AUC': f"{metrics['test_roc_auc']:.4f}",
            'Train_Accuracy': f"{metrics['train_accuracy']:.4f}",
            'Train_Recall': f"{metrics['train_recall']:.4f}",
            'Overfit_Gap': f"{metrics['overfitting_score']:.4f}",
            'Overfit_Level': metrics['overfitting_level'],
            'Time(s)': f"{metrics['training_time']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(output_dir, 'benchmark_comparison_table.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"  [OK] Saved: {csv_path}")
    
    # 2. Train vs Test Performance Comparison
    print("\n[VIZ 2/6] Train vs Test Performance (Overfitting Detection)")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    metrics_to_compare = [
        ('Accuracy', 'train_accuracy', 'test_accuracy'),
        ('Recall', 'train_recall', 'test_recall'),
        ('F1-Score', 'train_f1_score', 'test_f1_score'),
        ('ROC-AUC', 'train_roc_auc', 'test_roc_auc')
    ]
    
    for idx, (metric_name, train_key, test_key) in enumerate(metrics_to_compare):
        ax = axes[idx]
        model_names = list(results.keys())
        x = np.arange(len(model_names))
        width = 0.35
        
        train_values = [results[m][train_key] for m in model_names]
        test_values = [results[m][test_key] for m in model_names]
        
        bars1 = ax.bar(x - width/2, train_values, width, label='Train', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='#e74c3c', alpha=0.8)
        
        # Add gap annotations
        for i, model in enumerate(model_names):
            gap = train_values[i] - test_values[i]
            if gap > 0.05:  # Only annotate significant gaps
                ax.text(i, max(train_values[i], test_values[i]) + 0.02, 
                       f'Gap: {gap:.3f}', ha='center', fontsize=8, color='red')
        
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(f'{metric_name}: Train vs Test', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    train_test_path = os.path.join(output_dir, 'train_vs_test_comparison.png')
    plt.savefig(train_test_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {train_test_path}")
    
    # 3. ROC Curves Overlay
    print("\n[VIZ 3/6] ROC Curves Overlay")
    plt.figure(figsize=(10, 8))
    
    for model_name, metrics in results.items():
        # Calculate ROC curve using actual y_test
        fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
        
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['roc_auc']:.3f})",
                linewidth=2, color=metrics['color'])
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(output_dir, 'benchmark_roc_curves.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {roc_path}")
    
    # 4. Metrics Comparison Bar Charts
    print("\n[VIZ 4/6] Metrics Comparison Bar Charts")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'training_time']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Training Time (s)']
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        model_names = list(results.keys())
        values = [results[m][metric] for m in model_names]
        
        # Color the best performer in green
        best_idx = values.index(max(values)) if metric != 'training_time' else values.index(min(values))
        colors = ['#2ecc71' if i == best_idx else '#95a5a6' for i in range(len(values))]
        
        ax.barh(model_names, values, color=colors)
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(f'{label} Comparison', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v, i, f' {v:.3f}' if metric != 'training_time' else f' {v:.2f}s', 
                   va='center', fontsize=9)
    
    plt.tight_layout()
    metrics_path = os.path.join(output_dir, 'benchmark_metrics_comparison.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {metrics_path}")
    
    # 5. Confusion Matrices Grid
    print("\n[VIZ 5/6] Confusion Matrices Grid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    
    for idx, (model_name, metrics) in enumerate(results.items()):
        ax = axes[idx]
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   ax=ax, cbar=False, square=True)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['Stay', 'Leave'])
        ax.set_yticklabels(['Stay', 'Leave'])
    
    # Hide the last subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'model_confusion_matrices.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {cm_path}")
    
    # 6. Feature Importance Comparison
    print("\n[VIZ 6/6] Feature Importance Comparison")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Random Forest
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model_object']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[-10:]
        axes[0].barh(range(10), importances[indices], color='#2ecc71')
        axes[0].set_yticks(range(10))
        axes[0].set_yticklabels([feature_names[i] for i in indices], fontsize=9)
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Random Forest - Top 10 Features', fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
    
    # Decision Tree
    if 'Decision Tree' in results:
        dt_model = results['Decision Tree']['model_object']
        importances = dt_model.feature_importances_
        indices = np.argsort(importances)[-10:]
        axes[1].barh(range(10), importances[indices], color='#3498db')
        axes[1].set_yticks(range(10))
        axes[1].set_yticklabels([feature_names[i] for i in indices], fontsize=9)
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Decision Tree - Top 10 Features', fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
    
    # Logistic Regression
    if 'Logistic Regression' in results:
        lr_model = results['Logistic Regression']['model_object']
        importances = np.abs(lr_model.coef_[0])
        indices = np.argsort(importances)[-10:]
        axes[2].barh(range(10), importances[indices], color='#9b59b6')
        axes[2].set_yticks(range(10))
        axes[2].set_yticklabels([feature_names[i] for i in indices], fontsize=9)
        axes[2].set_xlabel('Absolute Coefficient')
        axes[2].set_title('Logistic Regression - Top 10 Features', fontweight='bold')
        axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    importance_path = os.path.join(output_dir, 'feature_importance_comparison.png')
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {importance_path}")


def select_best_model(results):
    """
    Select the best model using composite scoring.
    
    Criteria:
    - Recall (40%): Must catch leavers
    - ROC-AUC (30%): Overall discrimination
    - F1-Score (20%): Balance precision/recall
    - Training Time (10%): Production feasibility
    
    Parameters:
    -----------
    results : dict
        Results from all models
    
    Returns:
    --------
    tuple: (best_model_name, rankings_df)
    """
    print("\n" + "="*70)
    print("WINNER SELECTION - COMPOSITE SCORING")
    print("="*70)
    
    print("\nScoring Criteria:")
    print("  - Recall (Class 1):  40% weight (catch leavers)")
    print("  - ROC-AUC:          30% weight (discrimination)")
    print("  - F1-Score:         20% weight (balance)")
    print("  - Training Time:    10% weight (efficiency)")
    
    # Extract metrics
    model_names = list(results.keys())
    recalls = np.array([results[m]['recall'] for m in model_names])
    roc_aucs = np.array([results[m]['roc_auc'] for m in model_names])
    f1_scores = np.array([results[m]['f1_score'] for m in model_names])
    times = np.array([results[m]['training_time'] for m in model_names])
    
    # Normalize to 0-1 scale
    recall_norm = (recalls - recalls.min()) / (recalls.max() - recalls.min() + 1e-10)
    roc_norm = (roc_aucs - roc_aucs.min()) / (roc_aucs.max() - roc_aucs.min() + 1e-10)
    f1_norm = (f1_scores - f1_scores.min()) / (f1_scores.max() - f1_scores.min() + 1e-10)
    time_norm = (times - times.min()) / (times.max() - times.min() + 1e-10)
    
    # Calculate composite scores (lower time is better)
    composite_scores = (
        0.40 * recall_norm +
        0.30 * roc_norm +
        0.20 * f1_norm +
        0.10 * (1 - time_norm)
    )
    
    # Create rankings DataFrame
    rankings_data = []
    for i, model_name in enumerate(model_names):
        rankings_data.append({
            'Rank': 0,  # Will be filled after sorting
            'Model': model_name,
            'Composite Score': composite_scores[i],
            'Recall': results[model_name]['recall'],
            'ROC-AUC': results[model_name]['roc_auc'],
            'F1-Score': results[model_name]['f1_score'],
            'Time(s)': results[model_name]['training_time']
        })
    
    rankings_df = pd.DataFrame(rankings_data)
    rankings_df = rankings_df.sort_values('Composite Score', ascending=False).reset_index(drop=True)
    rankings_df['Rank'] = range(1, len(rankings_df) + 1)
    
    best_model_name = rankings_df.iloc[0]['Model']
    
    print("\n" + "="*70)
    print("FINAL RANKINGS")
    print("="*70)
    print(rankings_df.to_string(index=False))
    
    print("\n" + "="*70)
    print(f"WINNER: {best_model_name}")
    print("="*70)
    
    return best_model_name, rankings_df


def calculate_business_impact(results, y_test):
    """
    Calculate business impact and cost analysis.
    
    Assumptions:
    - Cost of False Negative (missed leaver): $50,000
    - Cost of False Positive (false alarm): $5,000
    
    Parameters:
    -----------
    results : dict
        Results from all models
    y_test : array-like
        Test target values
    
    Returns:
    --------
    pd.DataFrame: Business impact analysis
    """
    print("\n" + "="*70)
    print("BUSINESS IMPACT ANALYSIS")
    print("="*70)
    
    print("\nCost Assumptions:")
    print("  - False Negative (missed leaver): $50,000 (replacement cost)")
    print("  - False Positive (false alarm):   $5,000 (retention effort)")
    
    FN_COST = 50000
    FP_COST = 5000
    
    # Baseline: Do nothing (all False Negatives)
    num_leavers = y_test.sum()
    baseline_cost = num_leavers * FN_COST
    
    impact_data = []
    for model_name, metrics in results.items():
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        
        cost_fn = fn * FN_COST
        cost_fp = fp * FP_COST
        total_cost = cost_fn + cost_fp
        savings = baseline_cost - total_cost
        savings_pct = (savings / baseline_cost) * 100
        
        impact_data.append({
            'Model': model_name,
            'False Negatives': fn,
            'False Positives': fp,
            'FN Cost': f"${cost_fn:,}",
            'FP Cost': f"${cost_fp:,}",
            'Total Cost': f"${total_cost:,}",
            'Savings vs Baseline': f"${savings:,}",
            'Savings %': f"{savings_pct:.1f}%"
        })
    
    impact_df = pd.DataFrame(impact_data)
    
    print(f"\nBaseline Cost (do nothing): ${baseline_cost:,}")
    print("\n" + "-"*70)
    print(impact_df.to_string(index=False))
    
    return impact_df


def generate_benchmark_report(results, best_model_name, rankings_df, impact_df, output_dir):
    """
    Generate comprehensive text report with overfitting analysis.
    
    Parameters:
    -----------
    results : dict
        Results from all models
    best_model_name : str
        Name of winning model
    rankings_df : pd.DataFrame
        Rankings table
    impact_df : pd.DataFrame
        Business impact table
    output_dir : str
        Output directory
    """
    print("\n" + "="*70)
    print("GENERATING BENCHMARK REPORT (WITH OVERFITTING ANALYSIS)")
    print("="*70)
    
    report_path = os.path.join(output_dir, 'benchmark_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ML MODEL BENCHMARK - ATTRITION PREDICTION\n")
        f.write("WITH OVERFITTING DETECTION & PREVENTION ANALYSIS\n")
        f.write("="*70 + "\n")
        f.write(f"Dataset: 4,410 employees (16.12% attrition rate)\n")
        f.write(f"Train/Test Split: 80/20 (stratified)\n")
        f.write(f"Class Balancing: SMOTE (training set only)\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("PERFORMANCE SUMMARY (TEST SET)\n")
        f.write("="*70 + "\n\n")
        f.write(rankings_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("OVERFITTING ANALYSIS\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10} {'Level':<30}\n")
        f.write("-"*70 + "\n")
        for model_name, metrics in results.items():
            f.write(f"{model_name:<20} ")
            f.write(f"{metrics['train_accuracy']:>10.4f}  ")
            f.write(f"{metrics['test_accuracy']:>10.4f}  ")
            f.write(f"{metrics['overfitting_score']:>8.4f}  ")
            f.write(f"{metrics['overfitting_level']}\n")
        f.write("\n")
        f.write("Interpretation:\n")
        f.write("  - Gap < 0.02: Excellent (No overfitting)\n")
        f.write("  - Gap < 0.05: Good (Minimal overfitting)\n")
        f.write("  - Gap < 0.10: Moderate (Some overfitting)\n")
        f.write("  - Gap < 0.20: High (Significant overfitting)\n")
        f.write("  - Gap >= 0.20: Severe (Extreme overfitting)\n\n")
        
        f.write("="*70 + "\n")
        f.write(f"WINNER: {best_model_name}\n")
        f.write("="*70 + "\n")
        winner_metrics = results[best_model_name]
        f.write(f"Composite Score: {rankings_df.iloc[0]['Composite Score']:.4f}\n\n")
        f.write("Key Strengths:\n")
        f.write(f"  - Test Recall: {winner_metrics['test_recall']:.1%} of leavers caught\n")
        f.write(f"  - Test ROC-AUC: {winner_metrics['test_roc_auc']:.4f} (discrimination ability)\n")
        f.write(f"  - Test F1-Score: {winner_metrics['test_f1_score']:.4f} (balanced performance)\n")
        f.write(f"  - Training Time: {winner_metrics['training_time']:.2f} seconds\n")
        f.write(f"  - Overfitting Level: {winner_metrics['overfitting_level']}\n")
        f.write(f"  - Overfitting Score: {winner_metrics['overfitting_score']:.4f}\n\n")
        f.write("Recommended for: Production deployment\n\n")
        
        f.write("="*70 + "\n")
        f.write("BUSINESS IMPACT ANALYSIS\n")
        f.write("="*70 + "\n\n")
        f.write(impact_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*70 + "\n")
        f.write(f"1. Deploy {best_model_name} for production attrition prediction\n")
        
        # Add overfitting-specific recommendations
        worst_overfit = max(results.items(), key=lambda x: x[1]['overfitting_score'])
        if worst_overfit[1]['overfitting_score'] > 0.10:
            f.write(f"2. CAUTION: {worst_overfit[0]} shows significant overfitting (gap={worst_overfit[1]['overfitting_score']:.4f})\n")
            f.write(f"   Consider: regularization, pruning, or cross-validation\n")
        else:
            f.write("2. Overfitting is well-controlled across all models\n")
        
        f.write("3. Consider ensemble of top 3 models for improved robustness\n")
        f.write("4. Monitor model performance monthly for drift detection\n")
        f.write("5. Retrain quarterly with new employee data\n")
        f.write("6. Implement early warning system for high-risk employees\n")
        f.write("7. Conduct A/B testing of retention interventions\n")
        f.write("\n")
        f.write("="*70 + "\n")
    
    print(f"[OK] Saved comprehensive report: {report_path}")


def print_executive_summary(results, best_model_name, rankings_df, impact_df):
    """
    Print executive summary to console with overfitting analysis.
    
    Parameters:
    -----------
    results : dict
        Results from all models
    best_model_name : str
        Name of winning model
    rankings_df : pd.DataFrame
        Rankings table
    impact_df : pd.DataFrame
        Business impact table
    """
    print("\n" + "="*70)
    print("EXECUTIVE SUMMARY (WITH OVERFITTING ANALYSIS)")
    print("="*70)
    
    winner = results[best_model_name]
    
    print(f"\nBEST MODEL: {best_model_name}")
    print(f"  TEST SET PERFORMANCE:")
    print(f"    - Accuracy:  {winner['test_accuracy']:.2%}")
    print(f"    - Recall:    {winner['test_recall']:.2%} (catches {winner['test_recall']:.0%} of leavers)")
    print(f"    - Precision: {winner['test_precision']:.2%}")
    print(f"    - F1-Score:  {winner['test_f1_score']:.4f}")
    print(f"    - ROC-AUC:   {winner['test_roc_auc']:.4f}")
    
    print(f"\n  OVERFITTING ANALYSIS:")
    print(f"    - Train Accuracy: {winner['train_accuracy']:.2%}")
    print(f"    - Test Accuracy:  {winner['test_accuracy']:.2%}")
    print(f"    - Gap:            {winner['overfitting_score']:.4f}")
    print(f"    - Level:          {winner['overfitting_level']}")
    
    print(f"\nCONFUSION MATRIX (TEST SET):")
    tn, fp, fn, tp = winner['confusion_matrix'].ravel()
    print(f"  True Negatives:  {tn} (correct stay predictions)")
    print(f"  False Positives: {fp} (false alarms)")
    print(f"  False Negatives: {fn} (MISSED leavers)")
    print(f"  True Positives:  {tp} (correct leave predictions)")
    
    print(f"\nBUSINESS IMPACT:")
    best_impact = impact_df[impact_df['Model'] == best_model_name].iloc[0]
    print(f"  Estimated Savings: {best_impact['Savings vs Baseline']}")
    print(f"  Cost Reduction:    {best_impact['Savings %']}")
    
    # Overfitting summary for all models
    print(f"\nOVERFITTING SUMMARY (ALL MODELS):")
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['overfitting_score']):
        status = "✓" if metrics['overfitting_score'] < 0.05 else "⚠" if metrics['overfitting_score'] < 0.10 else "✗"
        print(f"  {status} {model_name:20s}: Gap = {metrics['overfitting_score']:.4f} ({metrics['overfitting_level']})")
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nGenerated Outputs:")
    print("  1. outputs/model_results/benchmark_comparison_table.csv")
    print("  2. outputs/model_results/train_vs_test_comparison.png  [NEW]")
    print("  3. outputs/model_results/benchmark_roc_curves.png")
    print("  4. outputs/model_results/benchmark_metrics_comparison.png")
    print("  5. outputs/model_results/model_confusion_matrices.png")
    print("  6. outputs/model_results/feature_importance_comparison.png")
    print("  7. outputs/model_results/benchmark_report.txt")
    print("\n" + "="*70)


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-MODEL BENCHMARK - ATTRITION PREDICTION")
    print("="*70)
    print("Comparing 5 Classification Algorithms")
    print("="*70)
    
    try:
        # Step 1: Load data
        print("\n[LOADING] Master attrition dataset...")
        df = pd.read_csv('outputs/master_attrition_data.csv')
        print(f"[OK] Loaded dataset: {df.shape}")
        
        # Step 2: Preprocess data
        (X_train_res, X_test, y_train_res, y_test, feature_names, scaler,
         train_dist_before, train_dist_after) = preprocess_data(df)
        
        # Step 3: Initialize models
        models = initialize_models()
        
        # Step 4: Train and evaluate all models
        results = benchmark_models(models, X_train_res, X_test, y_train_res, y_test)
        
        # Step 5: Generate visualizations
        generate_comparison_visualizations(results, feature_names, y_test, OUTPUT_DIR)
        
        # Step 6: Select best model
        best_model_name, rankings_df = select_best_model(results)
        
        # Step 7: Business impact analysis
        impact_df = calculate_business_impact(results, y_test)
        
        # Step 8: Generate comprehensive report
        generate_benchmark_report(results, best_model_name, rankings_df, impact_df, OUTPUT_DIR)
        
        # Step 9: Print executive summary
        print_executive_summary(results, best_model_name, rankings_df, impact_df)
        
    except Exception as e:
        print(f"\n[ERROR] Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

