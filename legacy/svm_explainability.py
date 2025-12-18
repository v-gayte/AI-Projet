"""
SVM Model Explainability Script using Permutation Importance
Explainable AI (XAI) Expert
Objective: Identify key factors influencing employee attrition predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

# ============================================================================
# 1. DATA PREPROCESSING (Identical to Benchmark)
# ============================================================================
print("=" * 80)
print("STEP 1: Data Preprocessing")
print("=" * 80)

# Get the directory where this script is located
script_dir = Path(__file__).parent
# Load the dataset
data_path = script_dir / "data" / "final_dataset.csv"
df = pd.read_csv(data_path)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Separate features (X) and target (y)
y = df['Attrition'].copy()
X = df.drop(columns=['Attrition'])

# Remove EmployeeID if present (not predictive)
if 'EmployeeID' in X.columns:
    X = X.drop(columns=['EmployeeID'])
    print("✓ EmployeeID column removed")

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")
print(f"Target distribution (%):\n{y.value_counts(normalize=True) * 100}")

# Encode categorical variables using get_dummies
X_encoded = pd.get_dummies(X, drop_first=True)
print(f"After encoding: {X_encoded.shape[1]} features")

# Split into Train/Test (80/20) with stratify and random_state
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train target distribution:\n{y_train.value_counts()}")
print(f"Test target distribution:\n{y_test.value_counts()}")

# Apply StandardScaler (CRITICAL for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to maintain column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("✓ StandardScaler applied to training and test sets")

# ============================================================================
# 2. RE-TRAIN CHAMPION MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Re-training Champion SVM Model")
print("=" * 80)

# Initialize SVM with winning parameters
svm_model = SVC(
    class_weight='balanced',
    kernel='rbf',
    probability=True,
    random_state=42
)

# Train on scaled data
svm_model.fit(X_train_scaled, y_train)
print("✓ SVM model trained on scaled data")

# Evaluate baseline performance
from sklearn.metrics import recall_score, accuracy_score
y_train_pred = svm_model.predict(X_train_scaled)
y_test_pred = svm_model.predict(X_test_scaled)

train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nModel Performance:")
print(f"  Train Recall: {train_recall:.4f}")
print(f"  Test Recall:  {test_recall:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# 3. PERMUTATION IMPORTANCE CALCULATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Permutation Importance Calculation")
print("=" * 80)

print("Calculating permutation importance on TEST set...")
print("This may take a few minutes (n_repeats=10)...")

# Calculate permutation importance on TEST set for better generalization
perm_importance = permutation_importance(
    svm_model,
    X_test_scaled,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring='recall',  # Business metric: detect departures
    n_jobs=-1  # Use all available cores
)

print("✓ Permutation importance calculated")

# ============================================================================
# 4. RESULTS PROCESSING AND VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Results Processing and Visualization")
print("=" * 80)

# Create DataFrame with importance scores
# Get feature names (after get_dummies, column names are preserved)
feature_names = X_train_scaled.columns.tolist()

# Create importance DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance_Mean': perm_importance.importances_mean,
    'Importance_Std': perm_importance.importances_std
})

# Sort by importance descending
importance_df = importance_df.sort_values('Importance_Mean', ascending=False).reset_index(drop=True)

print(f"\nTotal features analyzed: {len(importance_df)}")
print(f"\nTop 10 Most Important Factors for Attrition Prediction:")
print("=" * 80)

# Display Top 10
top10_df = importance_df.head(10)
for idx, row in top10_df.iterrows():
    print(f"{idx + 1:2d}. {row['Feature']:40s} | Importance: {row['Importance_Mean']:8.6f} (±{row['Importance_Std']:.6f})")

# Display as formatted table
print("\n" + "=" * 80)
print("Top 10 Factors Table:")
print("=" * 80)
display_columns = ['Feature', 'Importance_Mean', 'Importance_Std']
print(top10_df[display_columns].to_string(index=False))

# ============================================================================
# 5. VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Visualization")
print("=" * 80)

# Create horizontal bar chart for Top 10 factors
plt.figure(figsize=(12, 8))

# Prepare data for visualization (Top 10, sorted by importance)
plot_df = importance_df.head(10).sort_values('Importance_Mean', ascending=True)

# Create horizontal bar plot
sns.barplot(
    data=plot_df,
    y='Feature',
    x='Importance_Mean',
    palette='viridis',
    orient='h'
)

# Add error bars for standard deviation
plt.errorbar(
    plot_df['Importance_Mean'],
    range(len(plot_df)),
    xerr=plot_df['Importance_Std'],
    fmt='none',
    color='black',
    capsize=3,
    alpha=0.5
)

# Customize the plot
plt.xlabel('Permutation Importance (Recall-based)', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Top 10 Factors Influencing Employee Attrition\n(SVM Model - Permutation Importance)', 
          fontsize=14, fontweight='bold', pad=20)

# Add grid for better readability
plt.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (idx, row) in enumerate(plot_df.iterrows()):
    plt.text(row['Importance_Mean'] + 0.001, i, 
             f'{row["Importance_Mean"]:.4f}',
             va='center', fontsize=9, fontweight='bold')

plt.tight_layout()

# Save the plot
output_file = 'svm_feature_importance.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved as '{output_file}'")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETED!")
print("=" * 80)
print("\nKey Insights:")
print(f"- Most Important Factor: {importance_df.iloc[0]['Feature']} "
      f"(Importance: {importance_df.iloc[0]['Importance_Mean']:.6f})")
print(f"- Top 3 Factors:")
for i in range(min(3, len(importance_df))):
    print(f"  {i+1}. {importance_df.iloc[i]['Feature']}: "
          f"{importance_df.iloc[i]['Importance_Mean']:.6f}")

print("\n" + "=" * 80)
print("Interpretation:")
print("Higher importance values indicate that permuting (shuffling) that")
print("feature causes a larger drop in recall score, meaning the feature")
print("is more critical for predicting employee departures.")
print("=" * 80)

