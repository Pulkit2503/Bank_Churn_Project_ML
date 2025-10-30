# ML Training with Your Existing Bank Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# All 7 classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

print("🚀 TRAINING ON YOUR BANK CHURN DATASET")
print("=" * 50)

# 1. Load your existing data
df = pd.read_csv('bank_churn_data.csv')
print("📊 Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")

# 2. Identify target column
target_col = None
for col in ['churn', 'Exited', 'Churn']:
    if col in df.columns:
        target_col = col
        break

if not target_col:
    print("❌ No churn column found. Available columns:")
    print(df.columns.tolist())
    exit()

print(f"✅ Target variable: '{target_col}'")
print(f"📊 Churn rate: {(df[target_col].mean()*100):.2f}%")

# 3. Select features (use common bank features)
feature_candidates = [
    'CreditScore', 'credit_score', 'Age', 'age', 'Tenure', 'tenure',
    'Balance', 'balance', 'NumOfProducts', 'num_of_products', 'products_number',
    'HasCrCard', 'has_credit_card', 'credit_card', 
    'IsActiveMember', 'is_active_member', 'active_member',
    'EstimatedSalary', 'estimated_salary'
]

available_features = [col for col in feature_candidates if col in df.columns]

if len(available_features) < 3:
    print("❌ Not enough features found. Using all numerical columns...")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    available_features = [col for col in numerical_cols if col != target_col]

print(f"✅ Using {len(available_features)} features: {available_features}")

X = df[available_features]
y = df[target_col]

print(f"📊 Class distribution: {y.value_counts().to_dict()}")

# 4. Handle categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        print(f"🔧 Encoding categorical feature: {col}")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# 5. Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Data preprocessing completed")

# 6. Define 7 classifiers with hyperparameters
classifiers = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42),
        'params': {'C': [0.1, 1.0], 'solver': ['liblinear', 'lbfgs']}
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {'max_depth': [3, 5], 'min_samples_split': [2, 5]}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    },
    'Naive Bayes': {
        'model': GaussianNB(),
        'params': {'var_smoothing': [1e-9, 1e-8]}
    },
    'Support Vector Machine': {
        'model': SVC(random_state=42, probability=True),
        'params': {'C': [0.1, 1.0], 'kernel': ['linear', 'rbf']}
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2]}
    }
}

# 7. Train all models
print("\n🎯 TRAINING 7 CLASSIFIERS...")
results = {}

for name, config in classifiers.items():
    print(f"Training {name}...")
    
    if config['params']:
        grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        best_model = config['model']
        best_model.fit(X_train_scaled, y_train)
        best_params = "No hyperparameters"
    
    # Predictions and metrics
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc,
        'model': best_model,
        'best_params': best_params
    }
    
    print(f"   ✅ {name}: Accuracy = {accuracy:.4f}")

# 8. Compare results
print("\n" + "="*60)
print("📊 MODEL COMPARISON RESULTS")
print("="*60)

results_df = pd.DataFrame({
    'Algorithm': results.keys(),
    'Accuracy': [results[name]['accuracy'] for name in results.keys()],
    'Precision': [results[name]['precision'] for name in results.keys()],
    'Recall': [results[name]['recall'] for name in results.keys()],
    'F1-Score': [results[name]['f1_score'] for name in results.keys()],
    'AUC-Score': [results[name]['auc_score'] for name in results.keys()],
    'Best_Params': [str(results[name]['best_params'])[:30] + "..." for name in results.keys()]
}).sort_values('Accuracy', ascending=False)

print(results_df.to_string(index=False))

# 9. Find and save best model
best_model_name = results_df.iloc[0]['Algorithm']
best_accuracy = results_df.iloc[0]['Accuracy']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f}")
print(f"   F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")

joblib.dump(results[best_model_name]['model'], 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(available_features, 'feature_names.pkl')
joblib.dump(results_df, 'model_results.pkl')

print("💾 Models and results saved!")

# 10. Visualization
plt.figure(figsize=(15, 10))

# Accuracy comparison
plt.subplot(2, 2, 1)
values = [results[name]['accuracy'] for name in results_df['Algorithm']]
bars = plt.barh(results_df['Algorithm'], values, color='lightblue')
plt.xlabel('Accuracy Score')
plt.title('Model Accuracy Comparison')
plt.xlim(0, 1)
for bar, v in zip(bars, values):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{v:.3f}', va='center')

# F1-Score comparison
plt.subplot(2, 2, 2)
values = [results[name]['f1_score'] for name in results_df['Algorithm']]
bars = plt.barh(results_df['Algorithm'], values, color='lightgreen')
plt.xlabel('F1-Score')
plt.title('Model F1-Score Comparison')
plt.xlim(0, 1)
for bar, v in zip(bars, values):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{v:.3f}', va='center')

# Confusion matrix for best model
plt.subplot(2, 2, 3)
best_model = results[best_model_name]['model']
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Feature importance if available
plt.subplot(2, 2, 4)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=True)
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title('Feature Importance')
else:
    plt.text(0.5, 0.5, 'Feature Importance\nnot available\nfor this model', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Feature Importance')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")