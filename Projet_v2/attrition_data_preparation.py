"""
HR Attrition Prediction - Data Preparation Pipeline
====================================================
Production-ready script for pharmaceutical company employee attrition analysis.

Author: Senior Lead Data Scientist - HR Analytics
Purpose: Merge disparate data sources and engineer psychological features
         that explain WHY employees leave (Burnout, Stagnation, Loyalty).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = 'outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[OK] Created output directory: {OUTPUT_DIR}/")


def engineer_time_features(in_time_path, out_time_path):
    """
    STEP 1: TIME & ATTENDANCE ENGINEERING
    
    Business Logic: Employees working excessive hours are more prone to burnout.
    This function calculates average working hours per employee from raw time logs.
    
    Parameters:
    -----------
    in_time_path : str
        Path to CSV with clock-in times (columns = dates, rows = EmployeeIDs)
    out_time_path : str
        Path to CSV with clock-out times (columns = dates, rows = EmployeeIDs)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with EmployeeID and AverageWorkingHours
    """
    print("\n" + "="*70)
    print("STEP 1: TIME & ATTENDANCE FEATURE ENGINEERING")
    print("="*70)
    
    # Load time logs (first column is EmployeeID, stored as index)
    in_time_df = pd.read_csv(in_time_path, index_col=0)
    out_time_df = pd.read_csv(out_time_path, index_col=0)
    
    print(f"[OK] Loaded in_time.csv: {in_time_df.shape}")
    print(f"[OK] Loaded out_time.csv: {out_time_df.shape}")
    
    # Rename index to EmployeeID for clarity
    in_time_df.index.name = 'EmployeeID'
    out_time_df.index.name = 'EmployeeID'
    
    # Get date columns (all columns except EmployeeID which is now index)
    date_columns = in_time_df.columns.tolist()
    
    # Convert all date columns to datetime
    for col in date_columns:
        in_time_df[col] = pd.to_datetime(in_time_df[col], errors='coerce')
        out_time_df[col] = pd.to_datetime(out_time_df[col], errors='coerce')
    
    print(f"[OK] Converted {len(date_columns)} date columns to datetime format")
    
    # Calculate daily working duration in hours
    # Duration = (out_time - in_time) converted to hours
    duration_df = (out_time_df - in_time_df).apply(lambda x: x.dt.total_seconds() / 3600)
    
    # Compute average working hours per employee (ignoring NaNs)
    avg_working_hours = duration_df.mean(axis=1, skipna=True)
    
    # Create result dataframe
    time_features = pd.DataFrame({
        'EmployeeID': avg_working_hours.index,
        'AverageWorkingHours': avg_working_hours.values
    })
    
    print(f"[OK] Calculated AverageWorkingHours for {len(time_features)} employees")
    print(f"  - Mean working hours: {time_features['AverageWorkingHours'].mean():.2f}")
    print(f"  - Min working hours: {time_features['AverageWorkingHours'].min():.2f}")
    print(f"  - Max working hours: {time_features['AverageWorkingHours'].max():.2f}")
    
    return time_features


def merge_and_clean_data(general_path, manager_path, employee_path, time_features):
    """
    STEP 2: DATA INTEGRATION & CLEANING
    
    Business Logic: Merge all data sources into a single master table.
    Remove noise (zero-variance columns) and handle missing values intelligently.
    
    Parameters:
    -----------
    general_path : str
        Path to general_data.csv (demographics, target variable)
    manager_path : str
        Path to manager_survey_data.csv (job involvement, performance)
    employee_path : str
        Path to employee_survey_data.csv (satisfaction scores)
    time_features : pd.DataFrame
        Time features from Step 1
    
    Returns:
    --------
    pd.DataFrame
        Cleaned and merged master dataframe
    """
    print("\n" + "="*70)
    print("STEP 2: DATA MERGING & CLEANING")
    print("="*70)
    
    # Load all datasets
    general_df = pd.read_csv(general_path)
    manager_df = pd.read_csv(manager_path)
    employee_df = pd.read_csv(employee_path)
    
    print(f"[OK] Loaded general_data.csv: {general_df.shape}")
    print(f"[OK] Loaded manager_survey_data.csv: {manager_df.shape}")
    print(f"[OK] Loaded employee_survey_data.csv: {employee_df.shape}")
    
    # Sequential left joins on EmployeeID
    # Start with general data (contains target variable)
    df = general_df.copy()
    df = df.merge(manager_df, on='EmployeeID', how='left')
    df = df.merge(employee_df, on='EmployeeID', how='left')
    df = df.merge(time_features, on='EmployeeID', how='left')
    
    print(f"[OK] Merged all datasets: {df.shape}")
    
    # Drop zero-variance columns (noise removal)
    noise_columns = ['EmployeeCount', 'Over18', 'StandardHours']
    existing_noise = [col for col in noise_columns if col in df.columns]
    
    if existing_noise:
        df.drop(columns=existing_noise, inplace=True)
        print(f"[OK] Dropped {len(existing_noise)} zero-variance columns: {existing_noise}")
    
    # ROBUST IMPUTATION STRATEGY
    print("\nImputing missing values...")
    
    # Numeric skewed features: Use MEDIAN (resistant to outliers)
    numeric_skewed = ['NumCompaniesWorked', 'TotalWorkingYears']
    for col in numeric_skewed:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  - {col}: Filled {df[col].isnull().sum()} NaNs with median ({median_val:.2f})")
    
    # Categorical ordinal features: Use MODE (most frequent value)
    categorical_ordinal = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
    for col in categorical_ordinal:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else df[col].median()
            nan_count = df[col].isnull().sum()
            df[col].fillna(mode_val, inplace=True)
            print(f"  - {col}: Filled {nan_count} NaNs with mode ({mode_val})")
    
    print(f"\n[OK] Final cleaned dataset: {df.shape}")
    print(f"[OK] Remaining missing values: {df.isnull().sum().sum()}")
    
    return df


def create_psychological_features(df):
    """
    STEP 3: ADVANCED PSYCHOLOGICAL FEATURE ENGINEERING
    
    Business Logic: Create features that capture the PSYCHOLOGY behind attrition:
    - Burnout: Physical and mental exhaustion
    - Stagnation: Lack of career progression
    - Loyalty: Job-hopping behavior vs commitment
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned master dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with original + 9 new psychological features
    """
    print("\n" + "="*70)
    print("STEP 3: PSYCHOLOGICAL FEATURE ENGINEERING")
    print("="*70)
    
    df_copy = df.copy()
    
    # 1. OVERTIME_HOURS: Physical Strain Metric
    # Logic: Hours beyond standard 8-hour workday = burnout risk
    df_copy['Overtime_Hours'] = df_copy['AverageWorkingHours'] - 8
    print("[OK] [1/9] Overtime_Hours = AverageWorkingHours - 8")
    print(f"        Measures physical strain beyond standard workday")
    
    # 2. LOYALTY_RATIO: Job Hopper vs Career-Long Employee
    # Logic: Low ratio = frequent job changes, High ratio = company loyalty
    df_copy['Loyalty_Ratio'] = df_copy['YearsAtCompany'] / df_copy['TotalWorkingYears']
    df_copy['Loyalty_Ratio'].replace([np.inf, -np.inf], 0, inplace=True)
    df_copy['Loyalty_Ratio'].fillna(0, inplace=True)
    print("[OK] [2/9] Loyalty_Ratio = YearsAtCompany / TotalWorkingYears")
    print(f"        Detects job hoppers (low) vs loyalists (high)")
    
    # 3. PROMOTION_STAGNATION: Career Progression Frustration
    # Logic: High ratio = stuck without promotion = high frustration
    df_copy['Promotion_Stagnation'] = df_copy['YearsSinceLastPromotion'] / df_copy['YearsAtCompany']
    df_copy['Promotion_Stagnation'].replace([np.inf, -np.inf], 0, inplace=True)
    df_copy['Promotion_Stagnation'].fillna(0, inplace=True)
    print("[OK] [3/9] Promotion_Stagnation = YearsSinceLastPromotion / YearsAtCompany")
    print(f"        High values = frustration from lack of growth")
    
    # 4. MANAGER_STABILITY: Leadership Consistency
    # Logic: Consistent manager = stability, Frequent changes = disruption
    df_copy['Manager_Stability'] = df_copy['YearsWithCurrManager'] / df_copy['YearsAtCompany']
    df_copy['Manager_Stability'].replace([np.inf, -np.inf], 0, inplace=True)
    df_copy['Manager_Stability'].fillna(0, inplace=True)
    print("[OK] [4/9] Manager_Stability = YearsWithCurrManager / YearsAtCompany")
    print(f"        Measures impact of consistent leadership")
    
    # 5. PRIOR_TENURE_AVG: Previous Job Stability
    # Logic: Average time in previous companies = past loyalty pattern
    df_copy['Prior_Tenure_Avg'] = (df_copy['TotalWorkingYears'] - df_copy['YearsAtCompany']) / df_copy['NumCompaniesWorked']
    df_copy['Prior_Tenure_Avg'].replace([np.inf, -np.inf], 0, inplace=True)
    df_copy['Prior_Tenure_Avg'].fillna(0, inplace=True)
    print("[OK] [5/9] Prior_Tenure_Avg = (TotalWorkingYears - YearsAtCompany) / NumCompaniesWorked")
    print(f"        Average time spent in previous jobs")
    
    # 6. COMPA_RATIO_LEVEL: Compensation Fairness
    # Logic: Income relative to job level = pay equity perception
    df_copy['Compa_Ratio_Level'] = df_copy['MonthlyIncome'] / df_copy['JobLevel']
    print("[OK] [6/9] Compa_Ratio_Level = MonthlyIncome / JobLevel")
    print(f"        Is the pay fair for their rank?")
    
    # 7. HIKE_PER_PERFORMANCE: Reward-Effort Mismatch
    # Logic: Low ratio = high performance but low raise = demotivation
    df_copy['Hike_Per_Performance'] = df_copy['PercentSalaryHike'] / df_copy['PerformanceRating']
    print("[OK] [7/9] Hike_Per_Performance = PercentSalaryHike / PerformanceRating")
    print(f"        Reward vs Effort mismatch detection")
    
    # 8. AGE_WHEN_JOINED: Career Entry Stage
    # Logic: Young joiners = different expectations than senior hires
    df_copy['Age_When_Joined'] = df_copy['Age'] - df_copy['YearsAtCompany']
    print("[OK] [8/9] Age_When_Joined = Age - YearsAtCompany")
    print(f"        Career stage at hiring (junior vs senior)")
    
    # 9. BURNOUT_RISK_SCORE: Combined Physical + Mental Strain
    # Logic: High overtime + Poor work-life balance = MAXIMUM BURNOUT RISK
    # Note: Invert WorkLifeBalance so 1=worst, 4=best becomes 4=worst, 1=best
    df_copy['Burnout_Risk_Score'] = df_copy['Overtime_Hours'] * (5 - df_copy['WorkLifeBalance'])
    print("[OK] [9/9] Burnout_Risk_Score = Overtime_Hours * (5 - WorkLifeBalance)")
    print(f"        Combined metric: High overtime + Poor WLB = Max risk")
    
    print(f"\n[OK] Successfully created 9 psychological features")
    print(f"[OK] Dataset shape: {df_copy.shape}")
    
    return df_copy


def encode_target(df):
    """
    STEP 4: TARGET VARIABLE ENCODING
    
    Convert categorical Attrition (Yes/No) to binary integer for ML modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with Attrition column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with Attrition_Binary column added
    """
    print("\n" + "="*70)
    print("STEP 4: TARGET VARIABLE ENCODING")
    print("="*70)
    
    # Create binary target: Yes=1, No=0
    df['Attrition_Binary'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    attrition_counts = df['Attrition'].value_counts()
    attrition_rate = (attrition_counts.get('Yes', 0) / len(df)) * 100
    
    print(f"[OK] Created Attrition_Binary (1=Yes, 0=No)")
    print(f"  - Attrition Rate: {attrition_rate:.2f}%")
    print(f"  - Leavers (Yes): {attrition_counts.get('Yes', 0)}")
    print(f"  - Stayers (No): {attrition_counts.get('No', 0)}")
    
    return df


def generate_insights(df, output_dir):
    """
    STEP 5: VALIDATION & ANALYSIS
    
    Generate comprehensive validation outputs:
    - Console statistics (shape, correlations)
    - Correlation heatmap (psychological features vs target)
    - Boxplots (overtime and burnout by attrition status)
    - Text report (sorted correlations)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Final master dataframe with all features
    output_dir : str
        Directory to save outputs
    """
    print("\n" + "="*70)
    print("STEP 5: VALIDATION & ANALYSIS")
    print("="*70)
    
    # Console Statistics
    print(f"\n[STATS] FINAL DATASET SHAPE: {df.shape}")
    print(f"   - Total Employees: {df.shape[0]}")
    print(f"   - Total Features: {df.shape[1]}")
    
    # Missing values report
    missing_vals = df.isnull().sum()
    if missing_vals.sum() > 0:
        print(f"\n[WARNING] Missing Values:")
        print(missing_vals[missing_vals > 0])
    else:
        print(f"\n[OK] No missing values in final dataset")
    
    # Define psychological features
    psych_features = [
        'Overtime_Hours', 'Loyalty_Ratio', 'Promotion_Stagnation',
        'Manager_Stability', 'Prior_Tenure_Avg', 'Compa_Ratio_Level',
        'Hike_Per_Performance', 'Age_When_Joined', 'Burnout_Risk_Score'
    ]
    
    # Calculate correlations with target
    correlations = df[psych_features + ['Attrition_Binary']].corr()['Attrition_Binary'].drop('Attrition_Binary')
    correlations_sorted = correlations.abs().sort_values(ascending=False)
    
    print("\n" + "="*70)
    print("[ANALYSIS] FEATURE CORRELATIONS WITH ATTRITION")
    print("="*70)
    print("\nRanked by Absolute Correlation Strength:\n")
    for feature in correlations_sorted.index:
        corr_val = correlations[feature]
        direction = "[+] Increases" if corr_val > 0 else "[-] Decreases"
        print(f"  {feature:25s}: {corr_val:+.4f} {direction} attrition risk")
    
    # Save correlation report to text file
    report_path = os.path.join(output_dir, 'feature_correlations.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("HR ATTRITION PREDICTION - FEATURE CORRELATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Psychological Features Ranked by Correlation with Attrition:\n")
        f.write("-"*70 + "\n\n")
        for feature in correlations_sorted.index:
            corr_val = correlations[feature]
            f.write(f"{feature:30s}: {corr_val:+.4f}\n")
    print(f"\n[OK] Saved correlation report: {report_path}")
    
    # VISUALIZATION 1: Correlation Heatmap
    print("\n[VISUAL] Generating visualizations...")
    
    plt.figure(figsize=(12, 10))
    corr_matrix = df[psych_features + ['Attrition_Binary']].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Psychological Features - Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved correlation heatmap: {heatmap_path}")
    
    # VISUALIZATION 2: Attrition Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot 1: Overtime_Hours by Attrition
    sns.boxplot(
        data=df,
        x='Attrition',
        y='Overtime_Hours',
        palette={'Yes': '#e74c3c', 'No': '#2ecc71'},
        ax=axes[0]
    )
    axes[0].set_title('Overtime Hours by Attrition Status', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Attrition', fontsize=12)
    axes[0].set_ylabel('Overtime Hours', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Boxplot 2: Burnout_Risk_Score by Attrition
    sns.boxplot(
        data=df,
        x='Attrition',
        y='Burnout_Risk_Score',
        palette={'Yes': '#e74c3c', 'No': '#2ecc71'},
        ax=axes[1]
    )
    axes[1].set_title('Burnout Risk Score by Attrition Status', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Attrition', fontsize=12)
    axes[1].set_ylabel('Burnout Risk Score', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, 'attrition_boxplots.png')
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved attrition boxplots: {boxplot_path}")
    
    # Statistical validation
    print("\n" + "="*70)
    print("[STATS] STATISTICAL VALIDATION")
    print("="*70)
    
    for feature in ['Overtime_Hours', 'Burnout_Risk_Score']:
        leavers = df[df['Attrition'] == 'Yes'][feature]
        stayers = df[df['Attrition'] == 'No'][feature]
        
        print(f"\n{feature}:")
        print(f"  Leavers - Mean: {leavers.mean():.2f}, Median: {leavers.median():.2f}")
        print(f"  Stayers - Mean: {stayers.mean():.2f}, Median: {stayers.median():.2f}")
        print(f"  Difference: {leavers.mean() - stayers.mean():+.2f}")


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("HR ATTRITION PREDICTION - DATA PREPARATION PIPELINE")
    print("="*70)
    print("Pharmaceutical Company Employee Turnover Analysis")
    print("="*70)
    
    try:
        # STEP 1: Time & Attendance Features
        time_features = engineer_time_features(
            'data/in_time.csv',
            'data/out_time.csv'
        )
        
        # STEP 2: Merge & Clean Data
        master_df = merge_and_clean_data(
            'data/general_data.csv',
            'data/manager_survey_data.csv',
            'data/employee_survey_data.csv',
            time_features
        )
        
        # STEP 3: Psychological Feature Engineering
        master_df = create_psychological_features(master_df)
        
        # STEP 4: Target Encoding
        master_df = encode_target(master_df)
        
        # STEP 5: Save Final Dataset
        output_path = os.path.join(OUTPUT_DIR, 'master_attrition_data.csv')
        master_df.to_csv(output_path, index=False)
        print(f"\n[OK] Saved master dataset: {output_path}")
        
        # STEP 6: Generate Insights & Visualizations
        generate_insights(master_df, OUTPUT_DIR)
        
        # Success Summary
        print("\n" + "="*70)
        print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated Outputs:")
        print(f"  1. {os.path.join(OUTPUT_DIR, 'master_attrition_data.csv')}")
        print(f"  2. {os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')}")
        print(f"  3. {os.path.join(OUTPUT_DIR, 'attrition_boxplots.png')}")
        print(f"  4. {os.path.join(OUTPUT_DIR, 'feature_correlations.txt')}")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        raise

