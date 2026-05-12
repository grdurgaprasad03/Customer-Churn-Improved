import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# =========================
# LOAD FILES
# =========================

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))
importance_df = pickle.load(open('importance.pkl', 'rb'))
accuracy = pickle.load(open('accuracy.pkl', 'rb'))

# =========================
# TITLE
# =========================

st.title("📊 AI-Powered Customer Churn Prediction System")

st.markdown(
    "Predict whether a telecom customer is likely to churn using Machine Learning."
)

# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.header("📌 Customer Information")

# Numerical Inputs

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges",
    0.0,
    200.0,
    70.0
)

total_charges = st.sidebar.slider(
    "Total Charges",
    0.0,
    10000.0,
    1000.0
)

# Categorical Inputs

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No"]
)

# =========================
# CREATE INPUT DATAFRAME
# =========================

input_data = pd.DataFrame(columns=features)

# Fill all columns with 0
for col in features:
    input_data.loc[0, col] = 0

# =========================
# NUMERICAL FEATURES
# =========================

if 'tenure' in features:
    input_data['tenure'] = tenure

if 'MonthlyCharges' in features:
    input_data['MonthlyCharges'] = monthly_charges

if 'TotalCharges' in features:
    input_data['TotalCharges'] = total_charges

# =========================
# ENCODING CATEGORICAL FEATURES
# =========================

# Gender
if 'gender_Male' in features:
    input_data['gender_Male'] = 1 if gender == 'Male' else 0

# Contract
if 'Contract_One year' in features:
    input_data['Contract_One year'] = 1 if contract == 'One year' else 0

if 'Contract_Two year' in features:
    input_data['Contract_Two year'] = 1 if contract == 'Two year' else 0

# Internet Service
if 'InternetService_Fiber optic' in features:
    input_data['InternetService_Fiber optic'] = 1 if internet_service == 'Fiber optic' else 0

if 'InternetService_No' in features:
    input_data['InternetService_No'] = 1 if internet_service == 'No' else 0

# Payment Method
if 'PaymentMethod_Credit card (automatic)' in features:
    input_data['PaymentMethod_Credit card (automatic)'] = 1 if payment_method == 'Credit card (automatic)' else 0

if 'PaymentMethod_Electronic check' in features:
    input_data['PaymentMethod_Electronic check'] = 1 if payment_method == 'Electronic check' else 0

if 'PaymentMethod_Mailed check' in features:
    input_data['PaymentMethod_Mailed check'] = 1 if payment_method == 'Mailed check' else 0

# Paperless Billing
if 'PaperlessBilling_Yes' in features:
    input_data['PaperlessBilling_Yes'] = 1 if paperless_billing == 'Yes' else 0

# Tech Support
if 'TechSupport_Yes' in features:
    input_data['TechSupport_Yes'] = 1 if tech_support == 'Yes' else 0

# Online Security
if 'OnlineSecurity_Yes' in features:
    input_data['OnlineSecurity_Yes'] = 1 if online_security == 'Yes' else 0

# =========================
# SCALE INPUT DATA
# =========================

input_scaled = scaler.transform(input_data)

# =========================
# DASHBOARD METRICS
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model Accuracy",
        f"{accuracy:.2%}"
    )

with col2:
    st.metric(
        "Model Type",
        "XGBoost"
    )

with col3:
    st.metric(
        "Features Used",
        len(features)
    )

# =========================
# PREDICTION BUTTON
# =========================

if st.button("🔍 Predict Customer Churn"):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📈 Prediction Results")

    st.write(
        f"### Churn Probability: {probability:.2%}"
    )

    st.progress(float(probability))

    if prediction == 1:

        st.error("⚠️ Customer is likely to churn")

        st.subheader("💡 Recommended Retention Strategies")

        st.write("- Offer loyalty discount")
        st.write("- Provide long-term subscription plan")
        st.write("- Assign customer retention executive")
        st.write("- Offer premium customer support")

    else:

        st.success("✅ Customer is likely to stay")

        st.subheader("🎯 Customer Status")

        st.write(
            "This customer shows strong retention probability."
        )

# =========================
# FEATURE IMPORTANCE CHART
# =========================

st.subheader("📊 Top Feature Importance")

fig, ax = plt.subplots(figsize=(10, 6))

chart_data = importance_df.head(10)

ax.barh(
    chart_data['Feature'],
    chart_data['Importance']
)

ax.invert_yaxis()

st.pyplot(fig)

# =========================
# PROJECT INFORMATION
# =========================

st.subheader("📌 Project Information")

st.write(
    '''
    This project uses:
    - XGBoost Classifier
    - Scikit-learn
    - Streamlit
    - Feature Scaling
    - Machine Learning Classification
    - Real-time Prediction System
    '''
)