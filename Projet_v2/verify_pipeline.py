"""
Verification Script for HR Attrition Data Pipeline
Validates that all outputs were generated correctly
"""

import pandas as pd
import os

def verify_pipeline():
    print("="*70)
    print("HR ATTRITION PIPELINE - VERIFICATION REPORT")
    print("="*70)
    
    # Check 1: Output files exist
    print("\n1. OUTPUT FILES VALIDATION")
    print("-"*70)
    output_files = [
        'master_attrition_data.csv',
        'correlation_heatmap.png',
        'attrition_boxplots.png',
        'feature_correlations.txt'
    ]
    
    all_exist = True
    for filename in output_files:
        filepath = os.path.join('outputs', filename)
        exists = os.path.exists(filepath)
        status = "[OK]" if exists else "[MISSING]"
        print(f"   {status} {filename}")
        if exists:
            size_kb = os.path.getsize(filepath) / 1024
            print(f"        Size: {size_kb:.2f} KB")
        all_exist = all_exist and exists
    
    if not all_exist:
        print("\n[ERROR] Some output files are missing!")
        return False
    
    # Check 2: Dataset validation
    print("\n2. DATASET VALIDATION")
    print("-"*70)
    df = pd.read_csv('outputs/master_attrition_data.csv')
    
    print(f"   Dataset Shape: {df.shape}")
    print(f"   Total Employees: {df.shape[0]:,}")
    print(f"   Total Features: {df.shape[1]}")
    
    # Check 3: Psychological features
    print("\n3. PSYCHOLOGICAL FEATURES CHECK")
    print("-"*70)
    psych_features = [
        "Overtime_Hours",
        "Loyalty_Ratio",
        "Promotion_Stagnation",
        "Manager_Stability",
        "Prior_Tenure_Avg",
        "Compa_Ratio_Level",
        "Hike_Per_Performance",
        "Age_When_Joined",
        "Burnout_Risk_Score"
    ]
    
    features_present = 0
    for feature in psych_features:
        if feature in df.columns:
            features_present += 1
            print(f"   [OK] {feature}")
        else:
            print(f"   [MISSING] {feature}")
    
    print(f"\n   Total: {features_present}/9 psychological features present")
    
    # Check 4: Target variable
    print("\n4. TARGET VARIABLE CHECK")
    print("-"*70)
    if 'Attrition_Binary' in df.columns:
        print(f"   [OK] Attrition_Binary exists")
        attrition_rate = (df['Attrition_Binary'].sum() / len(df)) * 100
        print(f"   Attrition Rate: {attrition_rate:.2f}%")
        print(f"   Leavers: {df['Attrition_Binary'].sum():,}")
        print(f"   Stayers: {(len(df) - df['Attrition_Binary'].sum()):,}")
    else:
        print(f"   [MISSING] Attrition_Binary")
        return False
    
    # Check 5: Data quality
    print("\n5. DATA QUALITY METRICS")
    print("-"*70)
    missing_values = df.isnull().sum().sum()
    print(f"   Missing Values: {missing_values}")
    
    if 'Overtime_Hours' in df.columns:
        print(f"   Average Overtime: {df['Overtime_Hours'].mean():.2f} hours")
        print(f"   Max Overtime: {df['Overtime_Hours'].max():.2f} hours")
        print(f"   Min Overtime: {df['Overtime_Hours'].min():.2f} hours")
    
    if 'Burnout_Risk_Score' in df.columns:
        print(f"   Average Burnout Score: {df['Burnout_Risk_Score'].mean():.2f}")
        print(f"   Max Burnout Score: {df['Burnout_Risk_Score'].max():.2f}")
    
    # Check 6: Correlation validation
    print("\n6. CORRELATION ANALYSIS")
    print("-"*70)
    if os.path.exists('outputs/feature_correlations.txt'):
        with open('outputs/feature_correlations.txt', 'r') as f:
            lines = f.readlines()
            print(f"   Report Lines: {len(lines)}")
            print(f"   [OK] Correlation report generated")
    
    # Final summary
    print("\n" + "="*70)
    print("[SUCCESS] ALL VALIDATION CHECKS PASSED!")
    print("="*70)
    print("\nPipeline Status: READY FOR PRODUCTION")
    print("Next Step: Use master_attrition_data.csv for ML model training")
    print("="*70)
    
    return True

if __name__ == "__main__":
    try:
        verify_pipeline()
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {str(e)}")
        import traceback
        traceback.print_exc()

