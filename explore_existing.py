# Explore your existing bank_churn_data.csv
import pandas as pd
import numpy as np

print("🔍 EXPLORING YOUR EXISTING DATASET...")
print("=" * 50)

# Load the existing file
df = pd.read_csv('bank_churn_data.csv')

print(f"📊 Dataset Shape: {df.shape}")
print(f"📋 Columns: {list(df.columns)}")

print("\nFirst 5 rows:")
print(df.head())

# Check for churn column
churn_columns = ['churn', 'Exited', 'Churn']
target_col = None

for col in churn_columns:
    if col in df.columns:
        target_col = col
        break

if target_col:
    print(f"\n✅ Target variable: '{target_col}'")
    print(f"📈 Churn Rate: {(df[target_col].mean()*100):.2f}%")
    print(f"📊 Churn Distribution:\n{df[target_col].value_counts()}")
else:
    print("\n❌ No standard churn column found. Available columns:")
    for col in df.columns:
        print(f"  - {col}")

# Check data types
print(f"\n📊 Data Types:")
print(df.dtypes)

# Check for missing values
print(f"\n❓ Missing Values: {df.isnull().sum().sum()}")