# Find your dataset file
import os

print("🔍 SEARCHING FOR YOUR DATASET FILE...")
print("=" * 50)

# List all files in current directory
all_files = os.listdir()
csv_files = [f for f in all_files if f.endswith('.csv')]

print("📁 All CSV files in your folder:")
for file in csv_files:
    print(f"   - {file}")

if csv_files:
    print(f"\n✅ Found {len(csv_files)} CSV file(s)")
    print("Please use the exact filename in the next steps!")
else:
    print("\n❌ No CSV files found in the folder.")
    print("Make sure your dataset file is in the same folder as your Python files.")