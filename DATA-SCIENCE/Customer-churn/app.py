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

# ---------- Helper functions ---------- #
@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    """Load and preprocess Telco churn dataset."""
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
    """Train a RandomForest model on the dataset."""
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

# ---------- Sidebar UI ---------- #
st.sidebar.header("📝 Input Customer Profile")
with st.sidebar.expander("🔧 Configure Attributes", expanded=True):
    user_inputs = {}
    cols = st.columns(2)
    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            if feature == "gender_Male":
                gender = st.radio("Gender", ["Male", "Female"])
                user_inputs[feature] = 1 if gender == "Male" else 0
            elif feature in yn_features:
                label = feature.replace("_Yes", "").replace("_", " ").title()
                response = st.radio(f"{label}?", ["Yes", "No"], index=1)
                user_inputs[feature] = 1 if response == "Yes" else 0
            else:
                default = 12 if "tenure" in feature else 50.0 if "charges" in feature else 0.0
                user_inputs[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=0.0,
                    value=default,
                    step=1.0
                )

input_df = pd.DataFrame([user_inputs])
X_scaled = scaler.transform(input_df)

# ---------- Main Section ---------- #
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Predict the likelihood of a customer churning based on their subscription details.")

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("🔮 Prediction Result")
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    st.metric("Churn Likelihood", f"{probability * 100:.1f} %")
    if prediction == 0:
        st.success("✅ Customer likely to stay.")
    else:
        st.error("⚠️ High churn risk detected!")

    with st.expander("🧾 Customer Profile Summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

with col2:
    st.subheader("📊 Top Contributing Features")
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

# ---------- Data Exploration ---------- #
with st.expander("📂 View Training Dataset"):
    preview_df = load_dataset("DATA-SCIENCE/Customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    st.dataframe(preview_df.head(100), use_container_width=True)

# ---------- Footer or Bonus ---------- #
st.markdown("---")
st.caption("📈 Built with love using Streamlit + Scikit-Learn + Plotly | 📬 Contact: your_email@example.com")

