"""
HR data cleaning and merging script for predictive attrition modeling.
Data Scientist Expert - Pharmaceutical Company
"""

import pandas as pd
import numpy as np

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
DATA_DIR = "AI-Projet/data"

# ============================================================================
# 2. LOADING AND PREPROCESSING TIME DATA
# ============================================================================
print("=" * 80)
print("STEP 1: Loading and preprocessing time data")
print("=" * 80)

# Load time files
in_time = pd.read_csv(f"{DATA_DIR}/in_time.csv")
out_time = pd.read_csv(f"{DATA_DIR}/out_time.csv")

# Rename the first column (empty or "Unnamed: 0") to EmployeeID
first_col_in = in_time.columns[0]
first_col_out = out_time.columns[0]
in_time = in_time.rename(columns={first_col_in: "EmployeeID"})
out_time = out_time.rename(columns={first_col_out: "EmployeeID"})

# Set EmployeeID as index
in_time = in_time.set_index("EmployeeID")
out_time = out_time.set_index("EmployeeID")

# Convert all date columns to datetime
# Date columns are all columns except EmployeeID (which is now the index)
for col in in_time.columns:
    in_time[col] = pd.to_datetime(in_time[col], errors='coerce')
    
for col in out_time.columns:
    out_time[col] = pd.to_datetime(out_time[col], errors='coerce')

print(f"Number of employees in in_time: {len(in_time)}")
print(f"Number of employees in out_time: {len(out_time)}")
print(f"Number of date columns: {len(in_time.columns)}")

# ============================================================================
# 3. TIME FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Time feature engineering")
print("=" * 80)

# Ensure columns are in the same order
assert list(in_time.columns) == list(out_time.columns), "Columns must be in the same order"

# Calculate daily working duration (in hours)
# Iterate over each employee (index)
time_features_list = []

for emp_id in in_time.index:
    # Get arrival and departure times for this employee
    in_times = in_time.loc[emp_id]
    out_times = out_time.loc[emp_id]
    
    # Calculate durations (out_time - in_time) in hours
    # Ignore days where one of the two is NaN (absence)
    durations = []
    for date_col in in_time.columns:
        in_val = in_times[date_col]
        out_val = out_times[date_col]
        
        # If both values are present, calculate duration
        if pd.notna(in_val) and pd.notna(out_val):
            duration = (out_val - in_val).total_seconds() / 3600.0  # Convert to hours
            durations.append(duration)
    
    # Calculate aggregates
    if len(durations) > 0:
        durations_array = np.array(durations)
        mean_hours = np.mean(durations_array)
        std_hours = np.std(durations_array) if len(durations_array) > 1 else 0.0
        total_days = len(durations_array)
        overtime_days = np.sum(durations_array > 8.0)
        percent_overtime = (overtime_days / total_days) * 100.0 if total_days > 0 else 0.0
    else:
        # If no days worked, set default values
        mean_hours = 0.0
        std_hours = 0.0
        total_days = 0
        percent_overtime = 0.0
    
    time_features_list.append({
        'EmployeeID': emp_id,
        'MeanWorkingHours': mean_hours,
        'StdWorkingHours': std_hours,
        'PercentOvertime': percent_overtime,
        'TotalDaysWorked': total_days
    })

# Create time_features DataFrame indexed by EmployeeID
time_features = pd.DataFrame(time_features_list)
time_features = time_features.set_index("EmployeeID")

print(f"Time features created for {len(time_features)} employees")
print(f"Created columns: {list(time_features.columns)}")
print(f"\nTime features preview:")
print(time_features.head())

# ============================================================================
# 4. LOADING OTHER DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Loading other data")
print("=" * 80)

# Load main files
general_data = pd.read_csv(f"{DATA_DIR}/general_data.csv")
manager_survey = pd.read_csv(f"{DATA_DIR}/manager_survey_data.csv")
employee_survey = pd.read_csv(f"{DATA_DIR}/employee_survey_data.csv")

print(f"Number of employees in general_data: {len(general_data)}")
print(f"Number of employees in manager_survey: {len(manager_survey)}")
print(f"Number of employees in employee_survey: {len(employee_survey)}")

# ============================================================================
# 5. DATA MERGING (LEFT JOIN)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Data merging (LEFT JOIN)")
print("=" * 80)

# Use general_data as base
df = general_data.copy()

# Save initial number of rows
initial_rows = len(df)
print(f"Initial number of rows: {initial_rows}")

# Merge with manager_survey_data (LEFT JOIN on EmployeeID)
df = df.merge(manager_survey, on="EmployeeID", how="left")
print(f"After merging with manager_survey: {len(df)} rows")

# Merge with employee_survey_data (LEFT JOIN on EmployeeID)
df = df.merge(employee_survey, on="EmployeeID", how="left")
print(f"After merging with employee_survey: {len(df)} rows")

# Reset time_features index for merging
time_features_reset = time_features.reset_index()

# Merge with time_features (LEFT JOIN on EmployeeID)
df = df.merge(time_features_reset, on="EmployeeID", how="left")
print(f"After merging with time_features: {len(df)} rows")

# Verify that no rows were lost
assert len(df) == initial_rows, f"Error: {initial_rows - len(df)} rows lost during merging!"

print(f"✓ No rows lost. Final dataset: {len(df)} rows, {len(df.columns)} columns")

# ============================================================================
# 6. CLEANING AND SIMPLIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Cleaning and simplification")
print("=" * 80)

# Remove constant columns (only one unique value)
constant_cols = []
for col in df.columns:
    if df[col].nunique() == 1:
        constant_cols.append(col)

if constant_cols:
    print(f"Constant columns identified: {constant_cols}")
    df = df.drop(columns=constant_cols)
    print(f"✓ {len(constant_cols)} constant columns removed")
else:
    print("No constant columns found")

# Encode target Attrition (Yes/No -> 1/0)
if 'Attrition' in df.columns:
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    print(f"✓ Attrition encoded as binary (1=Yes, 0=No)")
    print(f"  Distribution: {df['Attrition'].value_counts().to_dict()}")
else:
    print("⚠ 'Attrition' column not found")

print(f"Dataset after cleaning: {len(df)} rows, {len(df.columns)} columns")

# ============================================================================
# 7. MISSING VALUE IMPUTATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: Missing value imputation")
print("=" * 80)

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove EmployeeID from lists if present
if 'EmployeeID' in numeric_cols:
    numeric_cols.remove('EmployeeID')
if 'EmployeeID' in categorical_cols:
    categorical_cols.remove('EmployeeID')

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# Check missing values before imputation
missing_before = df.isnull().sum().sum()
print(f"\nMissing values before imputation: {missing_before}")

if missing_before > 0:
    # Imputation for numeric variables (median)
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  ✓ {col} (numeric): {df[col].isnull().sum()} NaN replaced by median = {median_val}")
    
    # Imputation for categorical variables (mode)
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                mode_val = mode_val[0]
                df[col] = df[col].fillna(mode_val)
                print(f"  ✓ {col} (categorical): NaN replaced by mode = {mode_val}")
            else:
                # If no mode (all unique values), replace with default value
                df[col] = df[col].fillna("Unknown")
                print(f"  ✓ {col} (categorical): NaN replaced by 'Unknown' (no mode available)")
else:
    print("No missing values to impute")

# Verify that no missing values remain
missing_after = df.isnull().sum().sum()
print(f"\nMissing values after imputation: {missing_after}")

if missing_after > 0:
    print("⚠ WARNING: Missing values still remain!")
    print(df.isnull().sum()[df.isnull().sum() > 0])
else:
    print("✓ No missing values remaining")

# ============================================================================
# 8. FINAL VERIFICATIONS AND EXPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: Final verifications and export")
print("=" * 80)

# Display final dimensions
print(f"Final dataset dimensions: {df.shape[0]} rows × {df.shape[1]} columns")

# Display column list
print(f"\nColumn list ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:3d}. {col}")

# Verify that no missing values remain
missing_final = df.isnull().sum().sum()
print(f"\nFinal missing values check: {missing_final}")
if missing_final == 0:
    print("✓ No missing values detected")
else:
    print(f"⚠ WARNING: {missing_final} missing values detected")
    print(df.isnull().sum()[df.isnull().sum() > 0])

# Save final result (without index)
output_file = "final_dataset.csv"
df.to_csv(output_file, index=False)
print(f"\n✓ Final dataset saved to '{output_file}' (without index)")

print("\n" + "=" * 80)
print("PROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 80)
