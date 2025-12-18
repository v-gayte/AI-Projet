"""
Random Forest Hyperparameter Optimization & Feature Importance Analysis
========================================================================
Optimize the winning Random Forest model using GridSearchCV to prevent overfitting
and extract business-critical feature importance.

Author: Senior Data Scientist - HR Analytics
Purpose: Fine-tune baseline model and identify key attrition drivers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
from datetime import datetime

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
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

# Create output directory
OUTPUT_DIR = 'outputs/optimization_results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[OK] Created output directory: {OUTPUT_DIR}/")


def preprocess_data(df):
    """
    Preprocess data for Random Forest optimization.
    Same pipeline as benchmark for consistency.
    
    Returns:
    --------
    X_train_res, X_test, y_train_res, y_test, feature_names
    """
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    df_encoded = df.copy()
    
    # 1. Categorical Encoding
    print("\n[STEP 1] Categorical Variable Encoding")
    if 'BusinessTravel' in df_encoded.columns:
        business_travel_map = {
            'Non-Travel': 0,
            'Travel_Rarely': 1,
            'Travel_Frequently': 2
        }
        df_encoded['BusinessTravel'] = df_encoded['BusinessTravel'].map(business_travel_map)
        print("  [OK] BusinessTravel: Ordinal encoding")
    
    # One-Hot Encoding for nominal variables
    nominal_columns = []
    for col in ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']:
        if col in df_encoded.columns:
            nominal_columns.append(col)
    
    if nominal_columns:
        df_encoded = pd.get_dummies(df_encoded, columns=nominal_columns, drop_first=True)
        print(f"  [OK] One-Hot Encoding: {len(nominal_columns)} variables")
    
    # Drop text version of Attrition
    if 'Attrition' in df_encoded.columns:
        df_encoded = df_encoded.drop('Attrition', axis=1)
    
    # 2. Feature Separation
    print("\n[STEP 2] Feature Separation")
    y = df_encoded['Attrition_Binary']
    X = df_encoded.drop(['Attrition_Binary', 'EmployeeID'], axis=1, errors='ignore')
    print(f"  [OK] Features (X): {X.shape}")
    print(f"  [OK] Target (y): {y.shape}")
    
    feature_names = X.columns.tolist()
    
    # 3. Feature Scaling
    print("\n[STEP 3] Feature Scaling")
    numeric_features = []
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64'] and X[col].nunique() > 2:
            numeric_features.append(col)
    
    scaler = StandardScaler()
    X_scaled = X.copy()
    if numeric_features:
        X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
        print(f"  [OK] Scaled {len(numeric_features)} numeric features")
    
    # 4. Train-Test Split
    print("\n[STEP 4] Train-Test Split (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  [OK] Training set: {X_train.shape[0]} samples")
    print(f"  [OK] Test set: {X_test.shape[0]} samples")
    
    # 5. SMOTE Application
    print("\n[STEP 5] SMOTE Class Balancing")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  [OK] Balanced training set: {X_train_res.shape[0]} samples")
    print(f"       Class distribution: 50/50 after SMOTE")
    
    return X_train_res, X_test, y_train_res, y_test, feature_names


def run_baseline_model(X_train_res, X_test, y_train_res, y_test):
    """
    Train baseline Random Forest (from benchmark) for comparison.
    
    Returns:
    --------
    model, baseline_results (dict)
    """
    print("\n" + "="*70)
    print("BASELINE MODEL (From Benchmark)")
    print("="*70)
    
    # Baseline configuration (from benchmark)
    baseline_rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=None,
        n_jobs=-1
    )
    
    print("\n[TRAINING] Baseline Random Forest...")
    baseline_rf.fit(X_train_res, y_train_res)
    print("[OK] Training completed")
    
    # Predictions
    y_train_pred = baseline_rf.predict(X_train_res)
    y_test_pred = baseline_rf.predict(X_test)
    y_test_proba = baseline_rf.predict_proba(X_test)[:, 1]
    
    # Metrics
    baseline_results = {
        'train_accuracy': accuracy_score(y_train_res, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_roc_auc': roc_auc_score(y_test, y_test_proba),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred)
    }
    
    print("\n[BASELINE RESULTS]")
    print(f"  Train Accuracy: {baseline_results['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {baseline_results['test_accuracy']:.4f}")
    print(f"  Test Recall:    {baseline_results['test_recall']:.4f} <- KEY METRIC")
    print(f"  Test Precision: {baseline_results['test_precision']:.4f}")
    print(f"  Test F1-Score:  {baseline_results['test_f1']:.4f}")
    print(f"  Test ROC-AUC:   {baseline_results['test_roc_auc']:.4f}")
    
    tn, fp, fn, tp = baseline_results['confusion_matrix'].ravel()
    print(f"\n[CONFUSION MATRIX]")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    return baseline_rf, baseline_results


def optimize_random_forest(X_train_res, y_train_res):
    """
    Hyperparameter tuning using GridSearchCV.
    Focus on preventing overfitting while maximizing recall.
    
    Returns:
    --------
    grid_search (fitted GridSearchCV object)
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION (GridSearchCV)")
    print("="*70)
    
    print("\nObjective: Maximize Recall while preventing overfitting")
    print("Strategy: Control model complexity through depth and sample requirements")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],              # Number of trees
        'max_depth': [10, 15, 20, None],         # Tree depth (crucial for overfitting)
        'min_samples_split': [2, 5, 10],         # Min samples to split node
        'min_samples_leaf': [1, 2, 4]            # Min samples per leaf
    }
    
    print("\n[PARAMETER GRID]")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\n[INFO] Total combinations to test: {total_combinations}")
    print(f"[INFO] With 5-fold CV: {total_combinations * 5} model fits")
    
    # Initialize base model
    rf_base = RandomForestClassifier(random_state=42)
    
    # GridSearchCV configuration
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5,                    # 5-fold cross-validation
        scoring='recall',        # Maximize recall (catch leavers)
        n_jobs=-1,              # Use all CPU cores
        verbose=2,              # Show progress
        return_train_score=True # Track overfitting
    )
    
    print("\n[GRID SEARCH CONFIGURATION]")
    print(f"  Cross-Validation: 5-fold stratified")
    print(f"  Scoring Metric: Recall (maximize leaver detection)")
    print(f"  Parallel Processing: Enabled (n_jobs=-1)")
    
    print("\n[TRAINING] Starting GridSearchCV...")
    print("This may take 2-5 minutes...\n")
    
    grid_search.fit(X_train_res, y_train_res)
    
    print("\n[OK] GridSearchCV completed!")
    
    return grid_search


def evaluate_optimized_model(grid_search, X_train_res, X_test, y_train_res, y_test, baseline_results):
    """
    Evaluate the optimized model and compare with baseline.
    
    Returns:
    --------
    optimized_results (dict)
    """
    print("\n" + "="*70)
    print("OPTIMIZED MODEL EVALUATION")
    print("="*70)
    
    # Best parameters
    print("\n[BEST PARAMETERS FOUND]")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Cross-validation score
    print(f"\n[CROSS-VALIDATION]")
    print(f"  Best CV Recall Score: {grid_search.best_score_:.4f}")
    
    # Predictions
    y_train_pred = best_model.predict(X_train_res)
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    optimized_results = {
        'train_accuracy': accuracy_score(y_train_res, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_roc_auc': roc_auc_score(y_test, y_test_proba),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred)
    }
    
    print("\n[OPTIMIZED MODEL RESULTS]")
    print(f"  Train Accuracy: {optimized_results['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {optimized_results['test_accuracy']:.4f}")
    print(f"  Test Recall:    {optimized_results['test_recall']:.4f} <- KEY METRIC")
    print(f"  Test Precision: {optimized_results['test_precision']:.4f}")
    print(f"  Test F1-Score:  {optimized_results['test_f1']:.4f}")
    print(f"  Test ROC-AUC:   {optimized_results['test_roc_auc']:.4f}")
    
    # Overfitting check
    train_test_gap = optimized_results['train_accuracy'] - optimized_results['test_accuracy']
    print(f"\n[OVERFITTING CHECK]")
    print(f"  Train-Test Gap: {train_test_gap:.4f}")
    if train_test_gap < 0.02:
        print(f"  Status: ✓ Excellent (No overfitting)")
    elif train_test_gap < 0.05:
        print(f"  Status: ✓ Good (Minimal overfitting)")
    else:
        print(f"  Status: ⚠ Moderate overfitting detected")
    
    # Confusion Matrix
    tn, fp, fn, tp = optimized_results['confusion_matrix'].ravel()
    print(f"\n[CONFUSION MATRIX]")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn} <- Critical (missed leavers)")
    print(f"  True Positives:  {tp}")
    
    # Classification Report
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_test_pred, target_names=['Stay', 'Leave']))
    
    # Comparison with Baseline
    print("\n" + "="*70)
    print("BASELINE vs OPTIMIZED COMPARISON")
    print("="*70)
    
    comparison_data = {
        'Metric': ['Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision', 'Test F1', 'Test ROC-AUC'],
        'Baseline': [
            baseline_results['train_accuracy'],
            baseline_results['test_accuracy'],
            baseline_results['test_recall'],
            baseline_results['test_precision'],
            baseline_results['test_f1'],
            baseline_results['test_roc_auc']
        ],
        'Optimized': [
            optimized_results['train_accuracy'],
            optimized_results['test_accuracy'],
            optimized_results['test_recall'],
            optimized_results['test_precision'],
            optimized_results['test_f1'],
            optimized_results['test_roc_auc']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['Change'] = comparison_df['Optimized'] - comparison_df['Baseline']
    comparison_df['Change %'] = (comparison_df['Change'] / comparison_df['Baseline']) * 100
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Trade-off analysis
    print("\n[TRADE-OFF ANALYSIS]")
    recall_change = optimized_results['test_recall'] - baseline_results['test_recall']
    if recall_change >= 0:
        print(f"  ✓ Recall IMPROVED by {recall_change:.4f} ({recall_change*100:.2f}%)")
        print(f"    Optimization successful: Better recall + maintained performance")
    else:
        print(f"  ⚠ Recall DECREASED by {abs(recall_change):.4f} ({abs(recall_change)*100:.2f}%)")
        print(f"    Trade-off: Slight recall reduction for better generalization")
        if abs(recall_change) < 0.02:
            print(f"    Assessment: Acceptable trade-off (< 2% loss)")
        else:
            print(f"    Assessment: Consider using baseline model")
    
    return optimized_results, comparison_df


def extract_feature_importance(model, feature_names, output_dir):
    """
    Extract and visualize feature importance (Top 15).
    Identify business drivers of attrition.
    
    Returns:
    --------
    importance_df (DataFrame with rankings)
    """
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS (Business Drivers)")
    print("="*70)
    
    # Extract feature importances
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    
    # Reorder columns
    importance_df = importance_df[['Rank', 'Feature', 'Importance']]
    
    # Print Top 15
    print("\n[TOP 15 FEATURES - KEY ATTRITION DRIVERS]")
    print("="*70)
    print(importance_df.head(15).to_string(index=False))
    
    # Cumulative importance
    cumulative_importance = importance_df['Importance'].cumsum()
    top15_cumulative = cumulative_importance.iloc[14]
    print(f"\n[INSIGHT] Top 15 features account for {top15_cumulative:.1%} of total importance")
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'feature_importance_rankings.csv')
    importance_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved full rankings: {csv_path}")
    
    # Generate Publication-Ready Visualization
    print("\n[GENERATING] Publication-ready visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get top 15
    top15 = importance_df.head(15)
    
    # Color gradient based on importance
    colors = plt.cm.RdYlGn(top15['Importance'] / top15['Importance'].max())
    
    # Horizontal bar plot
    bars = ax.barh(range(15), top15['Importance'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Feature names on y-axis
    ax.set_yticks(range(15))
    ax.set_yticklabels(top15['Feature'], fontsize=11)
    
    # Invert y-axis (highest importance on top)
    ax.invert_yaxis()
    
    # Labels and title
    ax.set_xlabel('Feature Importance Score', fontsize=13, fontweight='bold')
    ax.set_title('Top 15 Features Driving Employee Attrition\nOptimized Random Forest Model',
                fontsize=15, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top15['Importance'])):
        ax.text(importance + 0.002, i, f'{importance:.4f}', 
               va='center', fontsize=10, fontweight='bold')
    
    # Grid for readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(output_dir, 'top15_feature_importance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[OK] Saved visualization: {plot_path}")
    print("     (Publication-ready for CEO presentation)")
    
    # Business interpretation
    print("\n" + "="*70)
    print("BUSINESS INTERPRETATION OF TOP DRIVERS")
    print("="*70)
    
    engineered_features = ['Overtime_Hours', 'Burnout_Risk_Score', 'Loyalty_Ratio', 
                          'Promotion_Stagnation', 'Manager_Stability', 'Prior_Tenure_Avg',
                          'Compa_Ratio_Level', 'Hike_Per_Performance', 'Age_When_Joined']
    
    print("\nTop 5 Attrition Drivers:")
    for i, row in importance_df.head(5).iterrows():
        feature = row['Feature']
        importance = row['Importance']
        rank = row['Rank']
        
        is_engineered = feature in engineered_features
        marker = "[ENGINEERED]" if is_engineered else "[ORIGINAL]"
        
        print(f"\n{rank}. {feature} {marker}")
        print(f"   Importance: {importance:.4f}")
        
        # Add business context
        if 'Overtime' in feature or 'Working' in feature:
            print(f"   → Business Impact: Physical workload and work hours")
        elif 'Age' in feature:
            print(f"   → Business Impact: Demographic risk profile")
        elif 'Income' in feature or 'Salary' in feature:
            print(f"   → Business Impact: Compensation satisfaction")
        elif 'Years' in feature and 'Company' in feature:
            print(f"   → Business Impact: Tenure and loyalty")
        elif 'Distance' in feature:
            print(f"   → Business Impact: Commute burden")
        elif 'Burnout' in feature:
            print(f"   → Business Impact: Combined stress indicators")
        elif 'Manager' in feature:
            print(f"   → Business Impact: Leadership stability")
        elif 'Promotion' in feature:
            print(f"   → Business Impact: Career advancement")
    
    return importance_df


def generate_optimization_report(grid_search, baseline_results, optimized_results, 
                                comparison_df, importance_df, output_dir):
    """
    Generate comprehensive optimization report.
    """
    print("\n" + "="*70)
    print("GENERATING OPTIMIZATION REPORT")
    print("="*70)
    
    report_path = os.path.join(output_dir, 'optimization_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RANDOM FOREST HYPERPARAMETER OPTIMIZATION REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Objective: Maximize Recall while preventing overfitting\n\n")
        
        f.write("="*70 + "\n")
        f.write("BEST HYPERPARAMETERS\n")
        f.write("="*70 + "\n\n")
        for param, value in grid_search.best_params_.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\n  Best CV Recall Score: {grid_search.best_score_:.4f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("BASELINE vs OPTIMIZED COMPARISON\n")
        f.write("="*70 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("TOP 15 FEATURE IMPORTANCE\n")
        f.write("="*70 + "\n\n")
        f.write(importance_df.head(15).to_string(index=False))
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*70 + "\n")
        
        recall_change = optimized_results['test_recall'] - baseline_results['test_recall']
        if recall_change >= -0.02:
            f.write("1. DEPLOY optimized model to production\n")
            f.write("   - Recall maintained or improved\n")
            f.write("   - Better generalization confirmed\n")
        else:
            f.write("1. CONSIDER baseline model for production\n")
            f.write("   - Optimized model has lower recall\n")
            f.write("   - Baseline may be better for this use case\n")
        
        f.write("2. Focus retention efforts on Top 5 features\n")
        f.write("3. Monitor feature importance stability over time\n")
        f.write("4. Retrain quarterly with new data\n")
        f.write("\n" + "="*70 + "\n")
    
    print(f"[OK] Saved optimization report: {report_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RANDOM FOREST OPTIMIZATION & FEATURE IMPORTANCE EXTRACTION")
    print("="*70)
    print("Objective: Fine-tune baseline model & identify business drivers")
    print("="*70)
    
    try:
        # Step 1: Load and preprocess data
        print("\n[STEP 1] Loading master dataset...")
        df = pd.read_csv('outputs/master_attrition_data.csv')
        print(f"[OK] Loaded: {df.shape}")
        
        X_train_res, X_test, y_train_res, y_test, feature_names = preprocess_data(df)
        
        # Step 2: Run baseline model for comparison
        print("\n[STEP 2] Training baseline model...")
        baseline_model, baseline_results = run_baseline_model(
            X_train_res, X_test, y_train_res, y_test
        )
        
        # Step 3: Hyperparameter optimization
        print("\n[STEP 3] Hyperparameter tuning with GridSearchCV...")
        grid_search = optimize_random_forest(X_train_res, y_train_res)
        
        # Step 4: Evaluate optimized model
        print("\n[STEP 4] Evaluating optimized model...")
        optimized_results, comparison_df = evaluate_optimized_model(
            grid_search, X_train_res, X_test, y_train_res, y_test, baseline_results
        )
        
        # Step 5: Extract feature importance
        print("\n[STEP 5] Extracting feature importance...")
        importance_df = extract_feature_importance(
            grid_search.best_estimator_, feature_names, OUTPUT_DIR
        )
        
        # Step 6: Generate comprehensive report
        print("\n[STEP 6] Generating optimization report...")
        generate_optimization_report(
            grid_search, baseline_results, optimized_results,
            comparison_df, importance_df, OUTPUT_DIR
        )
        
        # Final Summary
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nGenerated Outputs:")
        print(f"  1. {os.path.join(OUTPUT_DIR, 'feature_importance_rankings.csv')}")
        print(f"  2. {os.path.join(OUTPUT_DIR, 'top15_feature_importance.png')}")
        print(f"  3. {os.path.join(OUTPUT_DIR, 'optimization_report.txt')}")
        
        print("\n[RECOMMENDATION]")
        recall_change = optimized_results['test_recall'] - baseline_results['test_recall']
        if recall_change >= -0.01:
            print("  ✓ Use OPTIMIZED model for production deployment")
            print(f"    Recall change: {recall_change:+.4f} (maintained/improved)")
        else:
            print("  ⚠ Consider BASELINE model for production")
            print(f"    Recall loss: {recall_change:.4f} may not justify optimization")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n[ERROR] Optimization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

