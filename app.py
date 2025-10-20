import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

@st.cache_resource#to load these heavy objects only once

def load_resources():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    print("Resources loaded successfully.")
    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_resources()


st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("🏦 Customer Churn Prediction")
st.write("This app predicts whether a bank customer will churn (exit) based on their details. Please provide the customer's information below.")


st.subheader("Customer Demographics")
geography = st.selectbox("Geography", options=['France', 'Germany', 'Spain'])
gender = st.selectbox("Gender", options=['Male', 'Female'])
age = st.slider("Age", min_value=18, max_value=100, value=40)
    
st.subheader("Banking Information")
credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650)
tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card?", options=['Yes', 'No'])
is_active_member = st.selectbox("Is Active Member?", options=['Yes', 'No'])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0, step=5000.0)


if st.button("Predict Churn", type="primary"):

    input_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': 1 if has_cr_card == 'Yes' else 0,
        'IsActiveMember': 1 if is_active_member == 'Yes' else 0,
        'EstimatedSalary': estimated_salary
    }

    data = pd.DataFrame([input_data])
    st.write("---")
    st.subheader("Raw Input Data")
    st.dataframe(data)


    processed_data = data.copy()
    processed_data['Gender'] = label_encoder_gender.transform(processed_data['Gender'])

    geo_encoded = onehot_encoder_geo.transform(processed_data[['Geography']])
    geo_df = pd.DataFrame(geo_encoded.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    
    df = pd.concat([processed_data.drop('Geography', axis=1), geo_df], axis=1)

    expected_columns = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_France', 
        'Geography_Germany', 'Geography_Spain'
    ]
    df = df[expected_columns]

    scaled_df = scaler.transform(df)

    prediction_prob = model.predict(scaled_df)
    churn_probability = prediction_prob[0][0]

    st.subheader("✨ Prediction Result")
    
    if churn_probability > 0.5:
        st.error(f"**The customer is LIKELY to churn.** (Churn Probability: {churn_probability:.2%})")
        st.write("Consider taking retention actions for this customer.")
    else:
        st.success(f"**The customer is UNLIKELY to churn.** (Churn Probability: {churn_probability:.2%})")
        st.write("This customer seems loyal.")