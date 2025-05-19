import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from functools import lru_cache
import shap

st.set_page_config(
    page_title="📊 Telco Churn Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Helper functions ---------- #
@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    """Load and clean the Telco churn dataset"""
    df = pd.read_csv(path)
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    categorical_cols = df.select_dtypes("object").columns.tolist()
    if "Churn" in categorical_cols:
        categorical_cols.remove("Churn")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    X, y = df.drop("Churn", axis=1), df["Churn"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=400, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, X.columns

@lru_cache(maxsize=1)
def get_trained_components():
    df = load_dataset("DATA-SCIENCE/Customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return train_model(df)

@st.cache_resource
def get_explainer(model):
    return shap.TreeExplainer(model)

model, scaler, feature_names = get_trained_components()
explainer = get_explainer(model)

# Features originating from Yes/No columns all end with _Yes in one-hot encoding (except gender_Male handled separately)
yn_features = [f for f in feature_names if f.endswith("_Yes")]

# ---------- Sidebar input UI ---------- #
st.sidebar.header("📝 Input Customer Data")
with st.sidebar.expander("Fill in customer attributes", expanded=True):
    user_inputs = {}
    num_cols = 2
    cols = st.columns(num_cols)
    for idx, feature in enumerate(feature_names):
        with cols[idx % num_cols]:
            # Gender mapping 1 Male 2 Female
            if feature == "gender_Male":
                g_val = st.selectbox("Gender (1 = Male, 2 = Female)", [1, 2], index=0)
                user_inputs[feature] = 1 if g_val == 1 else 0
            # Generic Yes/No mapping 1 Yes 2 No
            elif feature in yn_features:
                label = feature.replace("_Yes", "").replace("_", " ").title()
                yn_val = st.selectbox(f"{label} (1=Yes, 2=No)", [1, 2], index=1)
                user_inputs[feature] = 1 if yn_val == 1 else 0
            else:
                default_val = 0.0
                if "tenure" in feature.lower():
                    default_val = 12
                elif "charges" in feature.lower():
                    default_val = 50.0
                user_inputs[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=0.0,
                    value=float(default_val),
                    step=1.0,
                    format="%.2f",
                )

input_df = pd.DataFrame([user_inputs])
X_scaled = scaler.transform(input_df)

# ---------- Prediction output ---------- #
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### 🔮 Prediction")
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]
    st.metric("Churn Likelihood", f"{prob * 100:.1f}%")
    st.success("Customer is likely to stay! 😊" if prediction == 0 else "High churn risk! ⚠️")

with col2:
    st.markdown("### 🏆 Top Feature Importances")
    importances = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    importances = importances.sort_values("Importance", ascending=False)
    friendly_names = (
        importances["Feature"]
        .str.replace("_Yes$", "", regex=True)
        .str.replace("_", " ")
        .str.title()
    )
    importances_plot = importances.copy()
    importances_plot["Friendly"] = friendly_names
    fig = px.bar(importances_plot.head(20), x="Importance", y="Friendly", orientation="h")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=500)
    st.plotly_chart(fig, use_container_width=True)

# ---------- SHAP Explanation ---------- #
st.markdown("### 🔍 Explain Prediction with SHAP")

# Calculate SHAP values for current input
shap_values = explainer.shap_values(X_scaled)[1]  # index 1 for positive class (Churn=1)
expected_value = explainer.expected_value[1]

# Create SHAP force plot using matplotlib and render in Streamlit
import matplotlib.pyplot as plt
import shap.plots._force as force

fig, ax = plt.subplots(figsize=(12, 3))
shap.force_plot(
    expected_value,
    shap_values[0],
    input_df.iloc[0],
    matplotlib=True,
    show=False,
    ax=ax
)
st.pyplot(fig)

# ---------- Dataset exploration ---------- #
with st.expander("📂 Peek at training dataset"):
    df_preview = load_dataset("DATA-SCIENCE/Customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    st.dataframe(df_preview.head(), use_container_width=True)
