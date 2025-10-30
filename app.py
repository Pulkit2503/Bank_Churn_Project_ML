# Streamlit App for Real Bank Data - FIXED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Real Bank Churn Predictor",
    page_icon="🏦",
    layout="centered"
)

@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('feature_names.pkl')
        results_df = joblib.load('model_results.pkl')
        return model, scaler, features, results_df
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, scaler, feature_names, results_df = load_models()

st.title("🏦 Indian Bank Customer Churn Prediction")
st.markdown("Using **Real Dataset** with 7 ML Algorithms | INR Currency")

# Sidebar - Only show if models loaded successfully
if results_df is not None:
    st.sidebar.header("📊 Model Performance")
    st.sidebar.dataframe(results_df[['Algorithm', 'Accuracy', 'F1-Score']].head(5))
    
    st.sidebar.header("🎯 Best Model Info")
    best_model_name = results_df.iloc[0]['Algorithm']
    best_accuracy = results_df.iloc[0]['Accuracy']
    best_f1 = results_df.iloc[0]['F1-Score']
    
    st.sidebar.write(f"**Algorithm:** {best_model_name}")
    st.sidebar.write(f"**Accuracy:** {best_accuracy:.4f}")
    st.sidebar.write(f"**F1-Score:** {best_f1:.4f}")
else:
    st.sidebar.warning("Models not loaded properly")

st.sidebar.header("ℹ️ Project Info")
st.sidebar.info("""
**7 Algorithms Used:**
- Logistic Regression
- K-Nearest Neighbors  
- Decision Tree
- Random Forest
- Naive Bayes
- SVM
- Gradient Boosting
""")

st.header("📊 Enter Customer Details (INR)")

# Create input fields based on available features
col1, col2 = st.columns(2)

input_data = {}

# Feature display names mapping
feature_display_names = {
    'CreditScore': 'Credit Score',
    'credit_score': 'Credit Score', 
    'Age': 'Age',
    'age': 'Age',
    'Tenure': 'Tenure (years)',
    'tenure': 'Tenure (years)',
    'Balance': 'Account Balance (₹)',
    'balance': 'Account Balance (₹)',
    'NumOfProducts': 'Number of Products',
    'num_of_products': 'Number of Products',
    'products_number': 'Number of Products',
    'HasCrCard': 'Has Credit Card?',
    'has_credit_card': 'Has Credit Card?',
    'credit_card': 'Has Credit Card?',
    'IsActiveMember': 'Is Active Member?', 
    'is_active_member': 'Is Active Member?',
    'active_member': 'Is Active Member?',
    'EstimatedSalary': 'Estimated Salary (₹)',
    'estimated_salary': 'Estimated Salary (₹)'
}

# Default values for features
default_values = {
    'CreditScore': 650, 'credit_score': 650,
    'Age': 45, 'age': 45,
    'Tenure': 5, 'tenure': 5,
    'Balance': 75000, 'balance': 75000,
    'NumOfProducts': 2, 'num_of_products': 2, 'products_number': 2,
    'HasCrCard': "Yes", 'has_credit_card': "Yes", 'credit_card': "Yes",
    'IsActiveMember': "Yes", 'is_active_member': "Yes", 'active_member': "Yes",
    'EstimatedSalary': 5000000, 'estimated_salary': 5000000
}

with col1:
    if feature_names:
        for i, feature in enumerate(feature_names):
            if i < len(feature_names) / 2:
                display_name = feature_display_names.get(feature, feature)
                default_val = default_values.get(feature, 0)
                
                if any(keyword in feature.lower() for keyword in ['credit', 'active', 'card', 'member']):
                    input_data[feature] = st.selectbox(display_name, ["No", "Yes"], index=1)
                elif 'age' in feature.lower():
                    input_data[feature] = st.slider(display_name, 18, 80, default_val)
                elif 'score' in feature.lower():
                    input_data[feature] = st.slider(display_name, 300, 850, default_val)
                elif 'tenure' in feature.lower():
                    input_data[feature] = st.slider(display_name, 0, 10, default_val)
                elif 'product' in feature.lower():
                    input_data[feature] = st.slider(display_name, 1, 4, default_val)
                elif 'balance' in feature.lower():
                    input_data[feature] = st.number_input(display_name, 0, 2500000, default_val)
                elif 'salary' in feature.lower():
                    input_data[feature] = st.number_input(display_name, 0, 20000000, default_val)

with col2:
    if feature_names:
        for i, feature in enumerate(feature_names):
            if i >= len(feature_names) / 2:
                display_name = feature_display_names.get(feature, feature)
                default_val = default_values.get(feature, 0)
                
                if any(keyword in feature.lower() for keyword in ['credit', 'active', 'card', 'member']):
                    input_data[feature] = st.selectbox(display_name, ["No", "Yes"], index=1)
                elif 'age' in feature.lower():
                    input_data[feature] = st.slider(display_name, 18, 80, default_val)
                elif 'score' in feature.lower():
                    input_data[feature] = st.slider(display_name, 300, 850, default_val)
                elif 'tenure' in feature.lower():
                    input_data[feature] = st.slider(display_name, 0, 10, default_val)
                elif 'product' in feature.lower():
                    input_data[feature] = st.slider(display_name, 1, 4, default_val)
                elif 'balance' in feature.lower():
                    input_data[feature] = st.number_input(display_name, 0, 2500000, default_val)
                elif 'salary' in feature.lower():
                    input_data[feature] = st.number_input(display_name, 0, 20000000, default_val)

# Convert categorical inputs to numerical
for feature in feature_names if feature_names else []:
    if feature in input_data and isinstance(input_data[feature], str):
        input_data[feature] = 1 if input_data[feature] == "Yes" else 0

# Prediction button
if st.button("🎯 Predict Churn Risk", type="primary"):
    if model is None or scaler is None or feature_names is None:
        st.error("❌ Models not loaded properly. Please check if all model files exist.")
    else:
        try:
            # Create feature array in correct order
            features_array = np.array([[input_data[feature] for feature in feature_names]])
            
            # Scale and predict
            features_scaled = scaler.transform(features_array)
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            st.header("📈 Prediction Results")
            
            # Display results in columns
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 1:
                    st.error("🚨 HIGH RISK: Customer will LEAVE")
                    st.metric("Churn Probability", f"{probability[1]*100:.2f}%")
                else:
                    st.success("✅ LOW RISK: Customer will STAY")
                    st.metric("Churn Probability", f"{probability[1]*100:.2f}%")
            
            with result_col2:
                if prediction == 1:
                    st.warning("**Recommendation:** Offer retention benefits, personal account manager, premium services")
                else:
                    st.info("**Recommendation:** Maintain current service, cross-sell products")
            
            # Probability chart
            st.subheader("Probability Distribution")
            chart_data = pd.DataFrame({
                'Status': ['Stay', 'Leave'],
                'Probability': [probability[0]*100, probability[1]*100]
            })
            st.bar_chart(chart_data.set_index('Status'))
            
            # Risk analysis
            st.subheader("🔍 Risk Analysis")
            if probability[1] > 0.7:
                st.error("**High Risk Category** - Immediate action required")
            elif probability[1] > 0.4:
                st.warning("**Medium Risk Category** - Monitor closely")
            else:
                st.success("**Low Risk Category** - Stable customer")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Indian Bank ML Project** | 7 Classification Algorithms | Hyperparameter Tuning | INR Currency")