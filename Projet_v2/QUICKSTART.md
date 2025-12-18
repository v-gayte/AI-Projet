# Quick Start Guide - HR Attrition Data Pipeline

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Pipeline
```bash
python attrition_data_preparation.py
```

### Step 3: Verify Results
```bash
python verify_pipeline.py
```

---

## 📂 What You'll Get

After running the pipeline, check the `outputs/` folder:

```
outputs/
├── master_attrition_data.csv    # ML-ready dataset (4,410 rows × 37 columns)
├── correlation_heatmap.png      # Feature correlation visualization
├── attrition_boxplots.png       # Distribution analysis by attrition status
└── feature_correlations.txt     # Sorted correlation report
```

---

## 📊 Expected Output

```
======================================================================
HR ATTRITION PREDICTION - DATA PREPARATION PIPELINE
======================================================================
Pharmaceutical Company Employee Turnover Analysis
======================================================================

STEP 1: TIME & ATTENDANCE FEATURE ENGINEERING
[OK] Loaded in_time.csv: (4410, 261)
[OK] Loaded out_time.csv: (4410, 261)
[OK] Calculated AverageWorkingHours for 4410 employees

STEP 2: DATA MERGING & CLEANING
[OK] Merged all datasets: (4410, 30)
[OK] Dropped 3 zero-variance columns
[OK] Final cleaned dataset: (4410, 27)

STEP 3: PSYCHOLOGICAL FEATURE ENGINEERING
[OK] [1/9] Overtime_Hours
[OK] [2/9] Loyalty_Ratio
[OK] [3/9] Promotion_Stagnation
[OK] [4/9] Manager_Stability
[OK] [5/9] Prior_Tenure_Avg
[OK] [6/9] Compa_Ratio_Level
[OK] [7/9] Hike_Per_Performance
[OK] [8/9] Age_When_Joined
[OK] [9/9] Burnout_Risk_Score

STEP 4: TARGET VARIABLE ENCODING
[OK] Created Attrition_Binary (1=Yes, 0=No)
  - Attrition Rate: 16.12%

STEP 5: VALIDATION & ANALYSIS
[ANALYSIS] FEATURE CORRELATIONS WITH ATTRITION
  Overtime_Hours           : +0.2017 [+] Increases attrition risk
  Burnout_Risk_Score       : +0.1920 [+] Increases attrition risk
  Manager_Stability        : -0.1307 [-] Decreases attrition risk

[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!
```

---

## 🎯 Key Features Created

| Feature | Formula | Business Logic |
|---------|---------|----------------|
| **Overtime_Hours** | `AverageWorkingHours - 8` | Physical strain beyond standard workday |
| **Burnout_Risk_Score** | `Overtime_Hours × (5 - WorkLifeBalance)` | Combined physical + mental strain |
| **Loyalty_Ratio** | `YearsAtCompany / TotalWorkingYears` | Job hopper vs loyalist indicator |
| **Promotion_Stagnation** | `YearsSinceLastPromotion / YearsAtCompany` | Career progression frustration |
| **Manager_Stability** | `YearsWithCurrManager / YearsAtCompany` | Leadership consistency impact |

---

## 📈 Key Insights

- **Attrition Rate**: 16.12% (711 out of 4,410 employees)
- **Top Driver**: Overtime Hours (correlation: +0.2017)
- **Leavers work 0.73 hours more per day** than stayers
- **Burnout Risk Score 169% higher** for leavers

---

## 🔧 Troubleshooting

### Issue: Module not found
```bash
# Solution: Install dependencies
pip install pandas numpy matplotlib seaborn
```

### Issue: File not found error
```bash
# Solution: Ensure you're in the project root directory
cd Projet_v2
python attrition_data_preparation.py
```

### Issue: Encoding error on Windows
**Already fixed!** The script automatically handles UTF-8 encoding.

---

## 📚 Documentation

- **README.md** - Full documentation with architecture and feature explanations
- **INSIGHTS.md** - Detailed analysis findings and strategic recommendations
- **QUICKSTART.md** - This file (quick reference)

---

## 🎓 Next Steps

1. **Explore the Data**
   ```python
   import pandas as pd
   df = pd.read_csv('outputs/master_attrition_data.csv')
   print(df.head())
   print(df.describe())
   ```

2. **Train a Model**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   
   # Load data
   df = pd.read_csv('outputs/master_attrition_data.csv')
   
   # Prepare features (exclude non-numeric and target)
   X = df.select_dtypes(include=['int64', 'float64']).drop(['Attrition_Binary', 'EmployeeID'], axis=1)
   y = df['Attrition_Binary']
   
   # Train model
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   
   # Evaluate
   print(f"Accuracy: {model.score(X_test, y_test):.2%}")
   ```

3. **Visualize Feature Importance**
   ```python
   import matplotlib.pyplot as plt
   
   # Get feature importance
   importances = model.feature_importances_
   features = X.columns
   
   # Plot
   plt.figure(figsize=(10, 6))
   plt.barh(features, importances)
   plt.xlabel('Importance')
   plt.title('Feature Importance for Attrition Prediction')
   plt.tight_layout()
   plt.savefig('outputs/feature_importance.png', dpi=300)
   ```

---

## ⏱️ Performance

- **Execution Time**: ~5-10 seconds
- **Memory Usage**: ~100 MB
- **Dataset Size**: 4,410 employees × 37 features
- **Output Size**: ~1.6 MB total

---

## 💼 Production Use

This pipeline is **production-ready** and includes:

✅ Error handling (division by zero, missing values)  
✅ Reproducibility (random seeds, deterministic operations)  
✅ Modularity (separate functions for each step)  
✅ Documentation (inline comments with business logic)  
✅ Validation (automated correlation analysis)  
✅ Cross-platform compatibility (Windows/Linux/Mac)

---

## 📞 Support

For questions or issues:
1. Check **README.md** for detailed documentation
2. Review **INSIGHTS.md** for analysis findings
3. Run `verify_pipeline.py` to diagnose issues

---

**Version**: 1.0.0  
**Last Updated**: December 2025  
**Author**: Senior Lead Data Scientist - HR Analytics

