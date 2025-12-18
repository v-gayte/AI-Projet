"""
Attrition Factors Visualization Script
Objective: Generate 5 specific visualizations to analyze key attrition factors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set seaborn theme
sns.set_theme(style="whitegrid")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 80)
print("Loading Data")
print("=" * 80)

# Get the directory where this script is located
script_dir = Path(__file__).parent
# Load the dataset
data_path = script_dir / "data" / "final_dataset.csv"
df = pd.read_csv(data_path)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Attrition distribution:\n{df['Attrition'].value_counts()}")
print(f"Attrition rate: {df['Attrition'].mean() * 100:.2f}%")

# Calculate global attrition rate for reference line
global_attrition_rate = df['Attrition'].mean()

# ============================================================================
# 2. GRAPH 1: PercentOvertime (Boxplot)
# ============================================================================
print("\n" + "=" * 80)
print("Creating Graph 1: Impact of Overtime Hours on Attrition")
print("=" * 80)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Attrition', y='PercentOvertime', palette=['lightblue', 'coral'])
plt.xlabel('Attrition (0 = Stay, 1 = Leave)', fontsize=12, fontweight='bold')
plt.ylabel('Percent Overtime (%)', fontsize=12, fontweight='bold')
plt.title('Impact of Overtime Hours on Employee Departure', fontsize=14, fontweight='bold', pad=15)
plt.xticks([0, 1], ['Stay (0)', 'Leave (1)'])

# Add statistics text
stay_median = df[df['Attrition'] == 0]['PercentOvertime'].median()
leave_median = df[df['Attrition'] == 1]['PercentOvertime'].median()
plt.text(0.5, plt.ylim()[1] * 0.95, 
         f'Median - Stay: {stay_median:.2f}% | Leave: {leave_median:.2f}%',
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('attrition_percent_overtime.png', dpi=300, bbox_inches='tight')
print("✓ Saved: attrition_percent_overtime.png")
plt.close()

# ============================================================================
# 3. GRAPH 2: TotalWorkingYears (Boxplot)
# ============================================================================
print("\n" + "=" * 80)
print("Creating Graph 2: Impact of Total Working Years on Attrition")
print("=" * 80)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Attrition', y='TotalWorkingYears', palette=['lightblue', 'coral'])
plt.xlabel('Attrition (0 = Stay, 1 = Leave)', fontsize=12, fontweight='bold')
plt.ylabel('Total Working Years', fontsize=12, fontweight='bold')
plt.title('Impact of Experience Level on Employee Departure\n(Juniors vs Seniors)', 
          fontsize=14, fontweight='bold', pad=15)
plt.xticks([0, 1], ['Stay (0)', 'Leave (1)'])

# Add statistics text
stay_median = df[df['Attrition'] == 0]['TotalWorkingYears'].median()
leave_median = df[df['Attrition'] == 1]['TotalWorkingYears'].median()
plt.text(0.5, plt.ylim()[1] * 0.95, 
         f'Median - Stay: {stay_median:.1f} years | Leave: {leave_median:.1f} years',
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('attrition_total_working_years.png', dpi=300, bbox_inches='tight')
print("✓ Saved: attrition_total_working_years.png")
plt.close()

# ============================================================================
# 4. GRAPH 3: YearsSinceLastPromotion (Boxplot)
# ============================================================================
print("\n" + "=" * 80)
print("Creating Graph 3: Impact of Career Stagnation on Attrition")
print("=" * 80)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Attrition', y='YearsSinceLastPromotion', palette=['lightblue', 'coral'])
plt.xlabel('Attrition (0 = Stay, 1 = Leave)', fontsize=12, fontweight='bold')
plt.ylabel('Years Since Last Promotion', fontsize=12, fontweight='bold')
plt.title('Impact of Career Stagnation on Employee Departure', 
          fontsize=14, fontweight='bold', pad=15)
plt.xticks([0, 1], ['Stay (0)', 'Leave (1)'])

# Add statistics text
stay_median = df[df['Attrition'] == 0]['YearsSinceLastPromotion'].median()
leave_median = df[df['Attrition'] == 1]['YearsSinceLastPromotion'].median()
plt.text(0.5, plt.ylim()[1] * 0.95, 
         f'Median - Stay: {stay_median:.1f} years | Leave: {leave_median:.1f} years',
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('attrition_years_since_promotion.png', dpi=300, bbox_inches='tight')
print("✓ Saved: attrition_years_since_promotion.png")
plt.close()

# ============================================================================
# 5. GRAPH 4: BusinessTravel (Barplot - Attrition Rate)
# ============================================================================
print("\n" + "=" * 80)
print("Creating Graph 4: Attrition Rate by Business Travel Frequency")
print("=" * 80)

# Calculate attrition rate by BusinessTravel category
travel_attrition = df.groupby('BusinessTravel')['Attrition'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=travel_attrition.index, y=travel_attrition.values, palette='viridis')
plt.xlabel('Business Travel Frequency', fontsize=12, fontweight='bold')
plt.ylabel('Attrition Rate', fontsize=12, fontweight='bold')
plt.title('Attrition Rate by Business Travel Frequency', fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=15, ha='right')

# Add reference line at global attrition rate (15% mentioned, but using actual rate)
plt.axhline(y=global_attrition_rate, color='red', linestyle='--', linewidth=2, 
            label=f'Global Average ({global_attrition_rate*100:.1f}%)')
plt.legend(loc='upper right', fontsize=10)

# Add value labels on bars
for i, (idx, val) in enumerate(travel_attrition.items()):
    plt.text(i, val + 0.01, f'{val*100:.1f}%', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('attrition_business_travel.png', dpi=300, bbox_inches='tight')
print("✓ Saved: attrition_business_travel.png")
plt.close()

# ============================================================================
# 6. GRAPH 5: JobRole (Barplot - Attrition Rate)
# ============================================================================
print("\n" + "=" * 80)
print("Creating Graph 5: Attrition Rate by Job Role")
print("=" * 80)

# Calculate attrition rate by JobRole category
jobrole_attrition = df.groupby('JobRole')['Attrition'].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 8))
sns.barplot(x=jobrole_attrition.index, y=jobrole_attrition.values, palette='viridis')
plt.xlabel('Job Role', fontsize=12, fontweight='bold')
plt.ylabel('Attrition Rate', fontsize=12, fontweight='bold')
plt.title('Attrition Rate by Job Role\n(Checking Research Directors)', 
          fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')

# Add reference line at global attrition rate
plt.axhline(y=global_attrition_rate, color='red', linestyle='--', linewidth=2, 
            label=f'Global Average ({global_attrition_rate*100:.1f}%)')
plt.legend(loc='upper right', fontsize=10)

# Add value labels on bars
for i, (idx, val) in enumerate(jobrole_attrition.items()):
    plt.text(i, val + 0.01, f'{val*100:.1f}%', 
             ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=0)

plt.tight_layout()
plt.savefig('attrition_job_role.png', dpi=300, bbox_inches='tight')
print("✓ Saved: attrition_job_role.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION COMPLETED!")
print("=" * 80)
print("\nGenerated 5 visualizations:")
print("  1. attrition_percent_overtime.png - Boxplot comparing overtime %")
print("  2. attrition_total_working_years.png - Boxplot comparing experience")
print("  3. attrition_years_since_promotion.png - Boxplot comparing stagnation")
print("  4. attrition_business_travel.png - Barplot of attrition by travel frequency")
print("  5. attrition_job_role.png - Barplot of attrition by job role")
print("\n" + "=" * 80)

# Display summary statistics
print("\nSummary Statistics:")
print("-" * 80)
print(f"Global Attrition Rate: {global_attrition_rate*100:.2f}%")
print("\nAttrition by Business Travel:")
for category, rate in travel_attrition.items():
    print(f"  {category}: {rate*100:.2f}%")
print("\nTop 3 Job Roles with Highest Attrition:")
for i, (role, rate) in enumerate(jobrole_attrition.head(3).items(), 1):
    print(f"  {i}. {role}: {rate*100:.2f}%")
print("\nTop 3 Job Roles with Lowest Attrition:")
for i, (role, rate) in enumerate(jobrole_attrition.tail(3).items(), 1):
    print(f"  {i}. {role}: {rate*100:.2f}%")


