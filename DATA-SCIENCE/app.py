import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your trained model and scaler (assumes you've saved them)
# For this example, we retrain quickly (in real app, load from disk)
@st.cache(allow_output_mutation=True)
def load_model():
    df = pd.read_csv('Customer-Churn.csv')
    
    # Drop irrelevant or broken data
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')  # Exclude target if in list

    # One-hot encode all categorical variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Fit the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model, X.columns.tolist()


model, feature_names = load_model()

st.title("Telco Customer Churn Prediction Dashboard")

st.sidebar.header("Input Customer Data")
inputs = {}
for feature in feature_names:
    inputs[feature] = st.sidebar.text_input(feature, "0")

input_df = pd.DataFrame([inputs])
input_df = input_df.astype(float)  # convert inputs to float for model

prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

st.subheader("Prediction")
st.write("Churn" if prediction==1 else "No Churn")
st.write(f"Probability of Churn: {probability:.2f}")

# Optional: Show feature importances
st.subheader("Feature Importances")
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
st.bar_chart(importance_df.set_index('Feature'))

