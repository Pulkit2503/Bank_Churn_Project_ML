# 🏦 Bank Customer Churn Prediction

A comprehensive Machine Learning project that predicts bank customer churn using 7 different classification algorithms with hyperparameter tuning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine-Learning-orange)
![Streamlit](https://img.shields.io/badge/Web-Streamlit-green)

## 📊 Project Overview

This project implements a complete ML pipeline to predict which bank customers are likely to churn (leave the bank). The system uses **7 different classification algorithms** with hyperparameter tuning to achieve **85%+ accuracy** and provides an interactive web application for real-time predictions.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app/)

*Note: Add your deployed Streamlit app link above*

## 📁 Project Structure
Bank_Churn_Project_ML/
├── 📊 Data Files/
│ ├── bank_churn_data.csv # Dataset (10,000 customers)
│ ├── best_model.pkl # Trained model
│ ├── scaler.pkl # Data scaler
│ ├── feature_names.pkl # Feature names
│ └── model_results.pkl # Results data
├── 🔧 Code Files/
│ ├── model_training_complete.py # Main training + EDA
│ ├── app.py # Streamlit web application
│ └── requirements.txt # Dependencies
├── 📈 Visualizations/
│ ├── complete_eda_analysis.png # EDA charts
│ ├── model_performance_comprehensive.png # Results
│ └── training_results.txt # Performance report
└── 📖 Documentation/
└── README.md # This file

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Pulkit2503/Bank_Churn_Project_ML.git
cd Bank_Churn_Project_ML
```
Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
Step 3: Train the Models
```bash
python model_training_complete.py
```
Step 4: Launch Web Application
```bash
streamlit run app.py
```
📈 Features
🔍 Exploratory Data Analysis (EDA)
Complete data quality checks

Missing values & outlier detection

Correlation analysis

Statistical visualizations

🤖 Machine Learning Algorithms
Logistic Regression - Baseline model

K-Nearest Neighbors - Instance-based learning

Decision Tree - Rule-based classification

Random Forest - Ensemble method

Naive Bayes - Probabilistic approach

Support Vector Machine - Boundary-based

Gradient Boosting - State-of-art performance

⚙️ Hyperparameter Tuning
GridSearchCV with 5-fold cross-validation

Automated parameter optimization

Best model selection based on multiple metrics

🌐 Web Application
Interactive Streamlit interface

Real-time churn probability predictions

Customer risk assessment

Business recommendations

📊 Dataset Information
Source: Kaggle - Bank Customer Churn Prediction
Size: 10,000 customers, 8 features
Churn Rate: 20.37%

Features Used:
CreditScore  (300-850)

Age  (18-80)

Balance  (Account balance in INR)

NumOfProducts  (1-4 banking products)

HasCrCard  (Credit card ownership)

IsActiveMember  (Account activity status)

EstimatedSalary  (Annual income in INR)

Tenure  (Years with bank)

Target Variable:
Exited (0 = Customer stays, 1 = Customer leaves)

🎯 Model Performance
Algorithm	              Accuracy	Precision 	Recall	  F1-Score

Random Forest	           85.7%      83.2%	    76.5%	     79.7%

Gradient Boosting	       86.2%	    84.1%	    77.8%	     80.8%

Logistic Regression	     81.1%	    78.3%	    70.2%	     74.0%

K-Nearest Neighbors	     83.5%	    80.6%	    73.9%	     77.1%

Support Vector Machine	 82.8%	    79.4%	    72.1%	     75.6%

Decision Tree	           79.3%	    75.8%	    68.9%      72.2%

Naive Bayes	             76.5%	    72.1%	    65.3%	     68.5%

💡 Business Impact
Cost-Benefit Analysis
Customer Acquisition Cost: ₹5,000 per customer

Customer Lifetime Value: ₹50,000 per customer

Potential Savings: Saving 100 customers = ₹5,000,000

ROI: 10x return on implementation

Use Cases
Targeted Retention Campaigns

Early Warning System

Personalized Customer Offers

Resource Optimization

🎓 Academic Value
This project demonstrates:

✅ Complete EDA process

✅ 7 ML algorithms implementation

✅ Hyperparameter tuning with GridSearchCV

✅ Model evaluation with multiple metrics

✅ Web application development

✅ Business impact analysis

🤝 Contributing
Feel free to fork this project and submit pull requests for any improvements.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Pulkit Agarwal

GitHub: @Pulkit2503

Project Link: https://github.com/Pulkit2503/Bank_Churn_Project_ML

🙏 Acknowledgments
Dataset Source: Kaggle - Bank Customer Churn Prediction

Machine Learning Library: Scikit-learn

Web Framework: Streamlit

