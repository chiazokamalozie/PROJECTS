import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from functools import lru_cache

st.set_page_config(
    page_title="📊 Telco Churn Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Helper Functions ---------- #
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
    model = RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, X.columns

@lru_cache(maxsize=1)
def get_trained_components():
    df = load_dataset("DATA-SCIENCE/Customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return train_model(df)

model, scaler, feature_names = get_trained_components()
yn_features = [f for f in feature_names if f.endswith("_Yes")]

# ---------- Sidebar Input UI ---------- #
st.sidebar.header("📝 Input Customer Data")
with st.sidebar.expander("Fill in customer attributes", expanded=True):
    user_inputs = {}
    cols = st.columns(2)
    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            label = feature.replace("_Yes", "").replace("_", " ").title()
            if feature == "":
                gender = st.radio("Gender", ["", ""])
                user_inputs[feature] = 1 if gender == "" else 0
            elif feature in yn_features:
                response = st.radio(f"{label}?", ["Yes", "No"], index=1)
                user_inputs[feature] = 1 if response == "Yes" else 0
            else:
                default = 12.0 if "tenure" in feature.lower() else 50.0 if "charges" in feature.lower() else 0.0
                min_val = float(0)
                step_val = float(1.0)
                user_inputs[feature] = st.number_input(
                    label,
                    min_value=min_val,
                    value=float(default),
                    step=step_val,
                    format="%.2f"
                )

input_df = pd.DataFrame([user_inputs])
X_scaled = scaler.transform(input_df)

# ---------- Main Area: Prediction + Visuals ---------- #
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Predict the likelihood of a customer leaving their telecom service based on various attributes.")

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("🔮 Prediction")
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    st.metric("Churn Likelihood", f"{probability * 100:.1f}%")
    if prediction == 0:
        st.success("✅ Customer likely to stay.")
    else:
        st.error("⚠️ High churn risk detected!")

    with st.expander("📋 Customer Input Summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

with col2:
    st.subheader("📊 Top Feature Importances")
    importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    importances["Label"] = (
        importances["Feature"]
        .str.replace("_Yes$", "", regex=True)
        .str.replace("_", " ")
        .str.title()
    )

    fig = px.bar(importances.head(15), x="Importance", y="Label", orientation="h",
                 color="Importance", color_continuous_scale="Blues", height=500)
    fig.update_layout(xaxis_title="Importance", yaxis_title="", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ---------- Dataset Explorer ---------- #
with st.expander("📂 Preview Training Dataset"):
    df_preview = load_dataset("DATA-SCIENCE/Customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    st.dataframe(df_preview.head(100), use_container_width=True)

# ---------- Footer ---------- #
st.markdown("---")
st.caption("Built with 💙 using Streamlit, Scikit-Learn, and Plotly | Contact: your_email@example.com")
