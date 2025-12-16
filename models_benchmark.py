"""
Model Benchmarking Script for Attrition Prediction
Lead Data Scientist - Risk Modeling Expert
Focus: Model stability and preventing overfitting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score

# ============================================================================
# 1. DATA PREPROCESSING
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

# Apply StandardScaler (MANDATORY for SVM and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to maintain column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("✓ StandardScaler applied to training and test sets")

# ============================================================================
# 2. MODEL CONFIGURATION (Anti-Overfitting Setup)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Model Configuration (Anti-Overfitting)")
print("=" * 80)

# Initialize models with conservative hyperparameters to prevent overfitting
models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced',
        solver='liblinear',
        penalty='l2',
        random_state=42,
        max_iter=1000
    ),
    'Random Forest': RandomForestClassifier(
        class_weight='balanced',
        n_estimators=100,
        max_depth=6,  # Limit depth to prevent memorization
        min_samples_leaf=5,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,  # Slow learning
        max_depth=3,  # Simple/weak trees
        random_state=42
    ),
    'SVM': SVC(
        class_weight='balanced',
        kernel='rbf',
        probability=True,
        random_state=42
    )
}

print("Models initialized with anti-overfitting hyperparameters:")
for name, model in models.items():
    print(f"  - {name}")

# ============================================================================
# 3. TRAINING AND EVALUATION LOOP
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Training and Evaluation")
print("=" * 80)

# Store results
results = []

for model_name, model in models.items():
    print(f"\n--- {model_name} ---")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    print("✓ Model trained")
    
    # Predict on training set
    y_train_pred = model.predict(X_train_scaled)
    
    # Predict on test set
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    
    # Calculate Overfitting Gap
    overfitting_gap = accuracy_train - accuracy_test
    
    print(f"  Accuracy (Train): {accuracy_train:.4f}")
    print(f"  Accuracy (Test):  {accuracy_test:.4f}")
    print(f"  Recall (Test):    {recall_test:.4f}")
    print(f"  Overfitting Gap:  {overfitting_gap:.4f}")
    
    if overfitting_gap > 0.05:
        print(f"  ⚠ OVERFITTING RISK DETECTED (Gap > 5%)")
    
    # Store results
    results.append({
        'Model': model_name,
        'Accuracy_Train': accuracy_train,
        'Accuracy_Test': accuracy_test,
        'Recall_Test': recall_test,
        'Overfitting_Gap': overfitting_gap
    })

# ============================================================================
# 4. RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Results Summary")
print("=" * 80)

# Create results DataFrame
results_df = pd.DataFrame(results)

# Add overfitting warning column
results_df['Overfitting_Warning'] = results_df['Overfitting_Gap'].apply(
    lambda x: '⚠ OVERFITTING RISK DETECTED' if x > 0.05 else ''
)

# Sort by Recall_Test (descending)
results_df = results_df.sort_values('Recall_Test', ascending=False).reset_index(drop=True)

# Display results
print("\nResults Summary (sorted by Recall_Test, descending):")
print("=" * 80)

# Format the display with warnings
for idx, row in results_df.iterrows():
    print(f"\n{idx + 1}. {row['Model']}")
    if row['Overfitting_Warning']:
        print(f"   {row['Overfitting_Warning']}")
    print(f"   Accuracy (Train): {row['Accuracy_Train']:.4f}")
    print(f"   Accuracy (Test):  {row['Accuracy_Test']:.4f}")
    print(f"   Recall (Test):     {row['Recall_Test']:.4f}")
    print(f"   Overfitting Gap:  {row['Overfitting_Gap']:.4f}")

# Display as DataFrame
print("\n" + "=" * 80)
print("Results Table:")
print("=" * 80)
display_df = results_df[['Model', 'Accuracy_Train', 'Accuracy_Test', 'Recall_Test', 'Overfitting_Gap', 'Overfitting_Warning']].copy()
print(display_df.to_string(index=False))

# ============================================================================
# 5. VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Visualization")
print("=" * 80)

# Prepare data for visualization
plot_data = results_df[['Model', 'Accuracy_Train', 'Accuracy_Test']].copy()
plot_data = plot_data.set_index('Model')

# Create grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(plot_data.index))
width = 0.35

bars1 = ax.bar(x - width/2, plot_data['Accuracy_Train'], width, 
               label='Train Accuracy', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, plot_data['Accuracy_Test'], width, 
               label='Test Accuracy', color='coral', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
ax.set_title('Model Comparison: Train vs Test Accuracy\n(Smaller Gap = More Stable Model)', 
              fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(plot_data.index, rotation=15, ha='right')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, max(plot_data.max()) * 1.15])

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'model_comparison.png'")
plt.show()

print("\n" + "=" * 80)
print("BENCHMARKING COMPLETED!")
print("=" * 80)
print("\nKey Insights:")
print(f"- Best Recall (Test): {results_df.iloc[0]['Model']} ({results_df.iloc[0]['Recall_Test']:.4f})")
print(f"- Most Stable Model (Smallest Gap): {results_df.loc[results_df['Overfitting_Gap'].idxmin(), 'Model']} "
      f"(Gap: {results_df['Overfitting_Gap'].min():.4f})")
print(f"- Models with Overfitting Risk: {len(results_df[results_df['Overfitting_Gap'] > 0.05])}")

