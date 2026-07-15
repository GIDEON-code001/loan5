from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics import mean_squared_error

MODEL_PATH = Path('loan_model.joblib')
FEATURES_PATH = Path('loan_preprocessor.joblib')

st.set_page_config(page_title='Loan Approval Predictor', page_icon='💳', layout='centered')
st.title('Loan Approval Prediction App')
st.write('Enter applicant details to predict whether the loan will be approved.')

try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
except FileNotFoundError:
    st.error('Model file not found. Train the model first by running train_model.py.')
    st.stop()

# Show evaluation metrics from the training data
try:
    df = pd.read_csv('loan.csv')
    X_eval = df[feature_columns]
    y_eval = df['Loan Approved'].astype(float)
    probs = model.predict_proba(X_eval)[:, 1]
    accuracy = model.score(X_eval, y_eval)
    rmse = np.sqrt(mean_squared_error(y_eval, probs))
    col1, col2 = st.columns(2)
    col1.metric('Accuracy', f'{accuracy:.2%}')
    col2.metric('RMSE', f'{rmse:.3f}')
except Exception:
    st.caption('RMSE and accuracy will be shown once the model and dataset are available.')

# Build form inputs
with st.form('loan_form'):
    gender = st.selectbox('Gender', ['Female', 'Male'])
    married = st.selectbox('Married', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    property_area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])
    home_ownership = st.selectbox('Home Ownership', ['Own', 'Mortgage', 'Rent'])
    loan_purpose = st.selectbox('Loan Purpose', ['Home', 'Education', 'Personal', 'Car', 'Business', 'Debt Consolidation'])
    age = st.number_input('Age', min_value=18, max_value=100, value=35)
    applicant_income = st.number_input('Applicant Income ($)', min_value=0, value=5000)
    coapplicant_income = st.number_input('Coapplicant Income ($)', min_value=0, value=1000)
    loan_amount = st.number_input('Loan Amount ($000)', min_value=0, value=100)
    loan_term = st.number_input('Loan Term (months)', min_value=1, max_value=360, value=360)
    credit_history = st.number_input('Credit History', min_value=0, max_value=1, value=1)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
    employment_years = st.number_input('Employment Years', min_value=0, value=5)
    existing_loans = st.number_input('Existing Loans', min_value=0, value=1)

    submitted = st.form_submit_button('Predict Approval')

if submitted:
    input_df = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self Employed': self_employed,
        'Property Area': property_area,
        'Home Ownership': home_ownership,
        'Loan Purpose': loan_purpose,
        'Age': age,
        'Applicant Income($)': applicant_income,
        'Coapplicant Income($)': coapplicant_income,
        'Loan Amount($000)': loan_amount,
        'Loan Term(months)': loan_term,
        'Credit History': credit_history,
        'Credit Score': credit_score,
        'Employment Years': employment_years,
        'Existing Loans': existing_loans,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f'Predicted Outcome: Approved with {probability * 100:.1f}% probability')
    else:
        st.error(f'Predicted Outcome: Rejected with {probability * 100:.1f}% probability')
