# Explore your downloaded dataset
import pandas as pd

# Load your dataset
df = pd.read_csv('bank_customer_churn_dataset.csv')  # or whatever the exact filename is

print("🔍 EXPLORING YOUR DATASET")
print("=" * 50)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Check the target variable
if 'churn' in df.columns:
    print(f"\n📊 Churn Distribution:")
    print(df['churn'].value_counts())
    print(f"Churn Rate: {(df['churn'].mean()*100):.2f}%")
else:
    print("\n❌ 'churn' column not found. Available columns:")
    for col in df.columns:
        print(f"  - {col}")