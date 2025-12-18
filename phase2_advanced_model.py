"""
Advanced ML Pipeline for Employee Attrition Prediction
Lead Data Scientist - Predictive HR Modeling Expert
Objective: Improve detection (Recall) and interpretability (SHAP) with business-driven features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_theme(style="whitegrid")

# ============================================================================
# 1. FEATURE ENGINEERING (Business Intelligence)
# ============================================================================
print("=" * 80)
print("STEP 1: Feature Engineering - Creating Business-Driven Features")
print("=" * 80)

# Get the directory where this script is located
script_dir = Path(__file__).parent
# Load the dataset
data_path = script_dir / "data" / "final_dataset.csv"
df = pd.read_csv(data_path)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Initial Attrition distribution:\n{df['Attrition'].value_counts()}")
print(f"Initial Attrition rate: {df['Attrition'].mean() * 100:.2f}%")

# Create new features
print("\nCreating new features...")

# 1. Income_to_Age_Ratio: Financial progression indicator
df['Income_to_Age_Ratio'] = df['MonthlyIncome'] / df['Age']
print("✓ Income_to_Age_Ratio created (MonthlyIncome / Age)")

# 2. Stability_Index: Average tenure per company
df['Stability_Index'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1)
print("✓ Stability_Index created (TotalWorkingYears / (NumCompaniesWorked + 1))")

# 3. Stress_Index: Sum of difficult conditions
df['Stress_Index'] = 0
df.loc[df['PercentOvertime'] > 15.0, 'Stress_Index'] += 1  # Overwork (>15%)
df.loc[df['BusinessTravel'] == 'Travel_Frequently', 'Stress_Index'] += 1  # Frequent travel
df.loc[df['DistanceFromHome'] > 20, 'Stress_Index'] += 1  # Long commute
df.loc[df['JobLevel'] == 1, 'Stress_Index'] += 1  # Junior under pressure
print("✓ Stress_Index created (sum of 4 stress conditions)")
print(f"  Stress_Index distribution:\n{df['Stress_Index'].value_counts().sort_index()}")

# 4. Seniority_Management: Group rare roles
df['Seniority_Management'] = df['JobRole'].copy()
senior_roles = ['Research Director', 'Manager', 'Manufacturing Director']
df.loc[df['JobRole'].isin(senior_roles), 'Seniority_Management'] = 'Senior_Mgmt'
print("✓ Seniority_Management created (grouped senior roles)")
print(f"  Senior_Mgmt count: {(df['Seniority_Management'] == 'Senior_Mgmt').sum()}")

# 5. Satisfaction_Gap: Performance vs happiness gap
df['Satisfaction_Gap'] = df['PerformanceRating'] - df['JobSatisfaction']
print("✓ Satisfaction_Gap created (PerformanceRating - JobSatisfaction)")
print(f"  Satisfaction_Gap range: [{df['Satisfaction_Gap'].min():.1f}, {df['Satisfaction_Gap'].max():.1f}]")

print(f"\nDataset after feature engineering: {df.shape[0]} rows, {df.shape[1]} columns")

# Display sample of new features
print("\nSample of new features (first 5 rows):")
new_features = ['Income_to_Age_Ratio', 'Stability_Index', 'Stress_Index', 
                'Seniority_Management', 'Satisfaction_Gap', 'Attrition']
print(df[new_features].head())

# ============================================================================
# 2. DATA PREPARATION & SMOTE (Class Balancing)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Data Preparation & SMOTE (Class Balancing)")
print("=" * 80)

# Separate features (X) and target (y)
y = df['Attrition'].copy()
X = df.drop(columns=['Attrition'])

# Remove EmployeeID if present (not predictive)
if 'EmployeeID' in X.columns:
    X = X.drop(columns=['EmployeeID'])
    print("✓ EmployeeID column removed")

# Drop the original JobRole column (we now use Seniority_Management)
if 'JobRole' in X.columns:
    X = X.drop(columns=['JobRole'])
    print("✓ JobRole column removed (replaced by Seniority_Management)")

print(f"Features shape before encoding: {X.shape}")

# Encode categorical variables using get_dummies
X_encoded = pd.get_dummies(X, drop_first=True)
print(f"Features shape after encoding: {X_encoded.shape[1]} features")

# Split into Train/Test (80/20) with stratify and random_state
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train target distribution BEFORE SMOTE:\n{y_train.value_counts()}")
print(f"Train attrition rate BEFORE SMOTE: {y_train.mean() * 100:.2f}%")

# Apply SMOTE only on training set (CRITICAL)
print("\nApplying SMOTE on training set only...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"✓ SMOTE applied")
print(f"Train set AFTER SMOTE: {X_train_smote.shape[0]} samples")
print(f"Train target distribution AFTER SMOTE:\n{pd.Series(y_train_smote).value_counts()}")
print(f"Train attrition rate AFTER SMOTE: {pd.Series(y_train_smote).mean() * 100:.2f}%")
print(f"Test set remains unchanged: {X_test.shape[0]} samples (REAL DATA)")

# Apply StandardScaler
print("\nApplying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_encoded.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_encoded.columns)

print("✓ StandardScaler applied to training and test sets")

# ============================================================================
# 3. MODELING (XGBoost)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: XGBoost Modeling")
print("=" * 80)

# Initialize XGBoost with recommended parameters
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)

print("Training XGBoost model...")
print(f"Parameters: n_estimators=200, learning_rate=0.05, max_depth=4")

# Train on SMOTE-balanced data
xgb_model.fit(X_train_scaled, y_train_smote, verbose=False)

print("✓ XGBoost model trained on SMOTE-balanced data")

# Predictions
y_train_pred = xgb_model.predict(X_train_scaled)
y_test_pred = xgb_model.predict(X_test_scaled)

# Calculate metrics
train_accuracy = accuracy_score(y_train_smote, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\nModel Performance:")
print(f"  Train Accuracy: {train_accuracy:.4f}")
print(f"  Test Accuracy:  {test_accuracy:.4f}")
print(f"  Test Precision: {test_precision:.4f}")
print(f"  Test Recall:    {test_recall:.4f} ← KEY METRIC (Departure Detection)")
print(f"  Test F1-Score:  {test_f1:.4f}")

# ============================================================================
# 4. INTERPRETABILITY (SHAP Values)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Interpretability with SHAP Values (MOST IMPORTANT FOR CLIENT)")
print("=" * 80)

print("Calculating SHAP values on test set...")
print("This may take a few minutes...")

# Create SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

print("✓ SHAP values calculated")

# Generate Beeswarm Plot
print("\nGenerating Beeswarm Plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_scaled, show=False)
plt.title('SHAP Beeswarm Plot - Feature Impact on Attrition\n(Red = High feature value, Blue = Low feature value)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap_beeswarm_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: shap_beeswarm_plot.png")
plt.close()

# Generate Bar Plot (Global Importance)
print("Generating Bar Plot (Global Importance)...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_scaled, plot_type='bar', show=False)
plt.title('SHAP Bar Plot - Global Feature Importance Ranking', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: shap_bar_plot.png")
plt.close()

# ============================================================================
# 5. METRICS & OUTPUT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Detailed Metrics & Output")
print("=" * 80)

# Classification Report
print("\nClassification Report (Test Set):")
print("=" * 80)
print(classification_report(y_test, y_test_pred, target_names=['Stay (0)', 'Leave (1)']))

# Confusion Matrix
print("\nConfusion Matrix (Test Set):")
print("=" * 80)
cm = confusion_matrix(y_test, y_test_pred)
print(f"                Predicted")
print(f"                Stay    Leave")
print(f"Actual Stay     {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"Actual Leave    {cm[1,0]:4d}    {cm[1,1]:4d}")
print()
print(f"True Negatives (Correctly predicted Stay):   {cm[0,0]}")
print(f"False Positives (Incorrectly predicted Leave): {cm[0,1]}")
print(f"False Negatives (Missed Departures):         {cm[1,0]} ← Critical to minimize!")
print(f"True Positives (Correctly detected Departures): {cm[1,1]}")

# Visualization of Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Stay (0)', 'Leave (1)'],
            yticklabels=['Stay (0)', 'Leave (1)'])
plt.xlabel('Predicted', fontsize=12, fontweight='bold')
plt.ylabel('Actual', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix - XGBoost Model on Test Set', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ADVANCED ML PIPELINE COMPLETED!")
print("=" * 80)

print("\nKey Results:")
print(f"- Test Recall (Departure Detection): {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"- Test Precision: {test_precision:.4f}")
print(f"- Test F1-Score: {test_f1:.4f}")
print(f"- Test Accuracy: {test_accuracy:.4f}")

print("\nNew Features Created:")
print("  1. Income_to_Age_Ratio - Financial progression")
print("  2. Stability_Index - Job stability")
print("  3. Stress_Index - Cumulative stress factors")
print("  4. Seniority_Management - Senior role grouping")
print("  5. Satisfaction_Gap - Performance vs happiness")

print("\nGenerated Files:")
print("  1. shap_beeswarm_plot.png - Feature impact visualization")
print("  2. shap_bar_plot.png - Global importance ranking")
print("  3. confusion_matrix.png - Prediction performance matrix")

print("\n" + "=" * 80)
print("Interpretation Guide:")
print("- Beeswarm Plot: Red dots (high values) on right → feature pushes toward departure")
print("- Beeswarm Plot: Blue dots (low values) on left → feature pushes toward retention")
print("- Bar Plot: Higher bars → more important features globally")
print("- Recall score shows % of actual departures we successfully detect")
print("=" * 80)

# Top 10 most important features based on mean absolute SHAP values
print("\nTop 10 Most Important Features (by mean |SHAP value|):")
print("=" * 80)
feature_importance = pd.DataFrame({
    'Feature': X_test_scaled.columns,
    'Mean_SHAP': np.abs(shap_values).mean(axis=0)
}).sort_values('Mean_SHAP', ascending=False).reset_index(drop=True)

for i, row in feature_importance.head(10).iterrows():
    print(f"{i+1:2d}. {row['Feature']:40s} | Mean |SHAP|: {row['Mean_SHAP']:.6f}")

