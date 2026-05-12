# 📊 AI-Powered Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn using XGBoost and Streamlit.

This project includes:
- Data preprocessing
- Feature engineering
- Machine Learning model training
- Model evaluation
- Feature importance analysis
- Interactive Streamlit dashboard
- Real-time churn prediction

---

# 🚀 Project Overview

Customer churn is one of the biggest challenges in telecom and subscription-based businesses.

This project helps businesses:
- Identify customers likely to churn
- Analyze important churn factors
- Take proactive retention actions
- Improve customer retention strategies

The system uses Machine Learning to predict churn probability based on customer demographics, billing details, and service usage patterns.

---

# 🛠️ Technologies Used

## Programming Language
- Python

## Libraries & Frameworks
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib

---

# 📁 Project Structure

```bash
Customer-Churn-Prediction/
│
├── data.py
├── main.py
├── churn.csv
├── model.pkl
├── scaler.pkl
├── features.pkl
├── importance.pkl
├── accuracy.pkl
├── requirements.txt
├── README.md
```

---

# 📌 Features

## ✅ Machine Learning Pipeline
- Data cleaning
- Missing value handling
- One-hot encoding
- Feature scaling
- Model training using XGBoost

## ✅ Model Evaluation
- Accuracy Score
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

## ✅ Interactive Dashboard
- Customer input form
- Real-time churn prediction
- Churn probability score
- Retention recommendations
- Feature importance visualization

## ✅ Business Insights
The application provides actionable retention strategies for high-risk customers.

---

# 📊 Model Information

## Model Used
XGBoost Classifier

## Why XGBoost?
XGBoost is a powerful gradient boosting algorithm known for:
- High prediction accuracy
- Speed and performance
- Handling structured/tabular data effectively

---

# 📈 Machine Learning Workflow

```text
Data Collection
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Train-Test Split
       ↓
Feature Scaling
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Deployment
       ↓
Real-Time Prediction
```

---

# ⚙️ Installation

## Step 1: Clone Repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
```

## Step 2: Navigate to Project Folder

```bash
cd customer-churn-prediction
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run The Project

## Step 1: Train the Model

```bash
python data.py
```

This generates:
- model.pkl
- scaler.pkl
- features.pkl
- importance.pkl
- accuracy.pkl

---

## Step 2: Run Streamlit App

```bash
streamlit run main.py
```

---

# 📷 Dashboard Features

## 📌 Customer Input Panel
Users can enter:
- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Internet Service
- Payment Method
- Tech Support
- Online Security

---

## 📌 Prediction Results
The dashboard displays:
- Churn prediction
- Churn probability
- Customer retention suggestions
- Risk analysis

---

## 📌 Feature Importance Chart
Visual representation of top factors affecting customer churn.

---

# 📊 Example Use Cases

- Telecom customer retention
- Subscription service analytics
- Customer risk analysis
- Business intelligence systems
- Predictive analytics applications

---

# 🎯 Business Impact

This system helps organizations:
- Reduce customer loss
- Improve customer satisfaction
- Increase retention rates
- Identify high-risk customers early
- Make data-driven retention decisions

---

# 📌 Future Improvements

Potential future enhancements:
- SHAP Explainability
- Deep Learning Models
- Cloud Deployment
- Batch Prediction
- REST API Integration
- Real-time Database Integration
- Docker Deployment
- MLOps Pipeline

---

# 👨‍💻 Author

Developed by [Durgaprasad G R]