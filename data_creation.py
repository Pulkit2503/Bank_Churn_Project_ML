# Create REALISTIC bank churn data with ACTUAL patterns
import pandas as pd
import numpy as np

print("📊 Creating REALISTIC Bank Churn Dataset with Actual Patterns...")

np.random.seed(42)
n_samples = 2000

# Create base features with correlations
data = {
    'CreditScore': np.random.normal(650, 100, n_samples).astype(int),
    'Age': np.random.normal(45, 15, n_samples).astype(int),
    'Tenure': np.random.randint(0, 11, n_samples),
    'Balance': np.random.exponential(500000, n_samples),
    'NumOfProducts': np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.5, 0.2, 0.1]),
    'HasCrCard': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'IsActiveMember': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    'EstimatedSalary': np.random.normal(5000000, 2000000, n_samples),
}

df = pd.DataFrame(data)

# Ensure realistic ranges
df['CreditScore'] = df['CreditScore'].clip(300, 850)
df['Age'] = df['Age'].clip(18, 80)
df['Balance'] = df['Balance'].clip(0, 2000000)
df['EstimatedSalary'] = df['EstimatedSalary'].clip(0, 15000000)

# Create STRONG, REALISTIC churn patterns
def calculate_churn_probability(row):
    probability = 0.0
    
    # Strong patterns
    if row['Age'] > 60:
        probability += 0.3  # Older customers more likely to churn
    if row['Balance'] < 10000:
        probability += 0.4  # Low balance = high churn
    if row['NumOfProducts'] == 1:
        probability += 0.2  # Single product users
    if row['IsActiveMember'] == 0:
        probability += 0.3  # Inactive members
    if row['CreditScore'] < 500:
        probability += 0.4  # Very low credit score
    if row['Tenure'] < 1:
        probability += 0.2  # Very new customers
    if row['EstimatedSalary'] > 8000000 and row['Balance'] < 50000:
        probability += 0.3  # High salary but low balance = might be leaving
    
    # Medium patterns
    if 45 < row['Age'] < 60:
        probability += 0.1
    if row['CreditScore'] < 600:
        probability += 0.15
    
    return probability

# Apply the function
churn_scores = df.apply(calculate_churn_probability, axis=1)

# Normalize and add some randomness
churn_scores = churn_scores / churn_scores.max()  # Normalize to 0-1
churn_scores += np.random.normal(0, 0.15, n_samples)  # Add noise
churn_scores = churn_scores.clip(0, 1)

# Convert to binary churn (0 or 1) - use threshold
df['Exited'] = (churn_scores > 0.35).astype(int)

print("✅ REALISTIC Dataset created with ACTUAL patterns!")
print(f"Dataset shape: {df.shape}")
print(f"Churn Rate: {(df['Exited'].mean()*100):.2f}%")

# Show feature correlations with churn
print("\n📊 Feature Correlations with Churn:")
correlations = df.corr()['Exited'].sort_values(ascending=False)
for feature, corr in correlations.items():
    if feature != 'Exited':
        print(f"   {feature}: {corr:.3f}")

# Save dataset
df.to_csv('bank_churn_data.csv', index=False)
print("\n💾 Dataset saved as 'bank_churn_data.csv'")

# Show pattern analysis
print("\n🔍 Pattern Analysis:")
print(f"Churn rate for Age > 60: {(df[df['Age'] > 60]['Exited'].mean()*100):.1f}%")
print(f"Churn rate for Balance < 10k: {(df[df['Balance'] < 10000]['Exited'].mean()*100):.1f}%")
print(f"Churn rate for Inactive members: {(df[df['IsActiveMember'] == 0]['Exited'].mean()*100):.1f}%")
print(f"Churn rate for Active members: {(df[df['IsActiveMember'] == 1]['Exited'].mean()*100):.1f}%")