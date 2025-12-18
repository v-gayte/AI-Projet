# Quick Start Guide - ML Model Benchmark

## 🚀 Run the Benchmark in 2 Steps

### Step 1: Install Dependencies (if not already done)
```bash
pip install -r requirements.txt
```

### Step 2: Run the Benchmark
```bash
python ml_model_benchmark.py
```

That's it! The benchmark will:
1. Load the prepared dataset (outputs/master_attrition_data.csv)
2. Preprocess data (encoding, scaling, SMOTE)
3. Train 5 models (Random Forest, Decision Tree, Logistic Regression, SVM, Perceptron)
4. Generate comprehensive comparison visualizations
5. Select the winner using composite scoring
6. Calculate business impact and savings

---

## 📊 Expected Output

### Console Output
```
======================================================================
MULTI-MODEL BENCHMARK - ATTRITION PREDICTION
======================================================================

[LOADING] Master attrition dataset...
[OK] Loaded dataset: (4410, 37)

PREPROCESSING PIPELINE
----------------------------------------------------------------------
[STEP 1] Categorical Variable Encoding
[STEP 2] Feature Separation
[STEP 3] Feature Scaling
[STEP 4] Train-Test Split
[STEP 5] SMOTE Class Balancing

MODEL TRAINING & EVALUATION
----------------------------------------------------------------------
[1/5] Random Forest
  [OK] Training completed in 0.23 seconds
  Recall: 0.9718 <- MOST IMPORTANT

[2/5] Decision Tree
[3/5] Logistic Regression
[4/5] SVM
[5/5] Perceptron

WINNER SELECTION
----------------------------------------------------------------------
WINNER: Random Forest

EXECUTIVE SUMMARY
----------------------------------------------------------------------
BEST MODEL: Random Forest
  - Accuracy:  99.55%
  - Recall:    97.18% (catches 97% of leavers)
  - Savings:   $6,900,000 (97.2% cost reduction)
```

### Generated Files

All saved to `outputs/model_results/`:

1. **benchmark_comparison_table.csv** - Metrics for all 5 models
2. **benchmark_roc_curves.png** - ROC curves overlay
3. **benchmark_metrics_comparison.png** - Bar charts (6 metrics)
4. **model_confusion_matrices.png** - Confusion matrix grid
5. **feature_importance_comparison.png** - Top 10 features
6. **benchmark_report.txt** - Comprehensive text report

---

## 🎯 Quick Results Summary

| Model | Accuracy | Recall | ROC-AUC | Winner? |
|-------|----------|--------|---------|---------|
| **Random Forest** | **99.55%** | **97.18%** | **0.9978** | ✅ 🏆 |
| SVM | 92.86% | 78.87% | 0.9579 | 🥈 |
| Decision Tree | 90.70% | 79.58% | 0.9319 | 🥉 |
| Logistic Regression | 81.86% | 58.45% | 0.7758 | |
| Perceptron | 71.77% | 52.11% | 0.7067 | |

**Winner: Random Forest** 
- Only misses 4 out of 142 leavers
- Zero false alarms
- $6.9M estimated annual savings

---

## 📈 View Results

### 1. Quick Metrics Table
```bash
# View comparison table
type outputs\model_results\benchmark_comparison_table.csv

# Or on Linux/Mac:
cat outputs/model_results/benchmark_comparison_table.csv
```

### 2. Detailed Report
```bash
# View full report
type outputs\model_results\benchmark_report.txt

# Or on Linux/Mac:
cat outputs/model_results/benchmark_report.txt
```

### 3. Visualizations
Open the PNG files in `outputs/model_results/`:
- ROC curves show Random Forest curve hugging top-left corner (optimal)
- Metrics comparison charts show Random Forest dominance in green
- Confusion matrices show Random Forest with minimal errors
- Feature importance validates engineered features

---

## 🔧 Customization Options

### Modify Model Configurations

Edit `ml_model_benchmark.py` to adjust hyperparameters:

```python
# Example: Increase Random Forest trees
'Random Forest': {
    'model': RandomForestClassifier(
        n_estimators=200,  # Changed from 100
        random_state=42,
        max_depth=None,
        n_jobs=-1
    ),
    ...
}
```

### Add New Models

```python
# Add to initialize_models() function
'XGBoost': {
    'model': XGBClassifier(
        n_estimators=100,
        random_state=42,
        use_label_encoder=False
    ),
    'description': 'Gradient Boosting classifier',
    'best_for': 'Structured data, high performance',
    'color': '#1abc9c'
}
```

### Change Scoring Weights

```python
# Edit select_best_model() function
composite_scores = (
    0.50 * recall_norm +     # Increase recall priority
    0.25 * roc_norm +        # Decrease ROC priority
    0.15 * f1_norm +
    0.10 * (1 - time_norm)
)
```

---

## 💡 Understanding the Metrics

### Recall (Most Important for Attrition)
- **What it measures**: Of all employees who actually left, how many did we catch?
- **Formula**: TP / (TP + FN)
- **Random Forest**: 97.18% - Only missed 4 out of 142 leavers
- **Why it matters**: Each missed leaver costs $50K in replacement

### Precision
- **What it measures**: Of all predicted leavers, how many actually left?
- **Formula**: TP / (TP + FP)
- **Random Forest**: 100% - Zero false alarms
- **Why it matters**: False alarms waste retention effort ($5K each)

### ROC-AUC
- **What it measures**: Model's ability to distinguish between classes
- **Range**: 0.5 (random) to 1.0 (perfect)
- **Random Forest**: 0.9978 - Near perfect discrimination
- **Why it matters**: Shows model confidence and reliability

### F1-Score
- **What it measures**: Harmonic mean of precision and recall
- **Range**: 0.0 to 1.0
- **Random Forest**: 0.9857 - Excellent balance
- **Why it matters**: Single metric for overall performance

---

## 🔍 Troubleshooting

### Issue: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'imblearn'
```

**Solution**: Install imbalanced-learn
```bash
pip install imbalanced-learn
```

### Issue: Low Memory
```
MemoryError: Unable to allocate array
```

**Solution**: Reduce n_estimators for Random Forest
```python
n_estimators=50  # Instead of 100
```

### Issue: SVM Takes Too Long
```
SVM training: > 30 seconds
```

**Solution**: Reduce training data or skip SVM
```python
# Comment out SVM in initialize_models()
# 'SVM': { ... }
```

---

## 📚 What You Get

### Performance Metrics
- ✅ Accuracy, Precision, Recall, F1-Score, ROC-AUC
- ✅ Confusion matrices for all models
- ✅ Training times (production feasibility)

### Business Metrics
- ✅ Cost analysis (FN @ $50K, FP @ $5K)
- ✅ Estimated annual savings per model
- ✅ ROI calculations

### Visualizations
- ✅ ROC curves overlay (all models on one chart)
- ✅ Metrics comparison bar charts (6 metrics)
- ✅ Confusion matrix heatmap grid
- ✅ Feature importance side-by-side

### Winner Selection
- ✅ Composite scoring (40% recall, 30% ROC-AUC, 20% F1, 10% time)
- ✅ Final rankings table
- ✅ Justification for winner selection

---

## 🎓 Next Steps After Benchmark

### 1. Deploy Winner to Production
```python
import joblib
from ml_model_benchmark import preprocess_data

# Load the model (you'll need to save it first)
model = joblib.load('models/random_forest_attrition.pkl')

# Score new employees
new_employees_df = pd.read_csv('new_employees.csv')
X_new, _, _, _, _, _ = preprocess_data(new_employees_df)
predictions = model.predict_proba(X_new)[:, 1]

# Flag high-risk employees
high_risk = new_employees_df[predictions > 0.5]
```

### 2. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='recall'
)
```

### 3. Feature Engineering Round 2
Based on feature importance, create new features:
- Interaction terms (e.g., Overtime × Distance)
- Polynomial features
- Time-based aggregations

### 4. Ensemble Methods
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('rf', random_forest_model),
        ('svm', svm_model),
        ('dt', decision_tree_model)
    ],
    voting='soft'
)
```

---

## ⏱️ Benchmark Performance

**Total Execution Time**: ~7-10 seconds

| Step | Time |
|------|------|
| Data Loading | 0.1s |
| Preprocessing | 0.5s |
| Random Forest | 0.2s |
| Decision Tree | 0.1s |
| Logistic Regression | 0.1s |
| SVM | 6.8s |
| Perceptron | 0.01s |
| Visualizations | 1.5s |
| **Total** | **~9.2s** |

---

## 💼 Business Value

Using Random Forest:
- **Catch 97% of leavers** (138 out of 142)
- **Zero false alarms** (no wasted retention effort)
- **Save $6.9M annually** (vs doing nothing)
- **97.2% cost reduction**
- **ROI: 13,660%**

---

## 📞 Support

For questions or issues:
1. Check `ML_BENCHMARK_SUMMARY.md` for detailed analysis
2. Review `outputs/model_results/benchmark_report.txt`
3. Verify all dependencies installed: `pip list | grep -E "sklearn|imbalanced|pandas"`

---

**Version**: 1.0.0  
**Last Updated**: December 2025  
**Author**: Lead Data Scientist - HR Analytics

