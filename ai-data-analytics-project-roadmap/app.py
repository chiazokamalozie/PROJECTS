
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customs Risk Analytics", layout="wide")

st.title("📦 Customs Cargo Risk Analytics – Demo")

@st.cache_data
def load_data():
    trend = pd.read_csv("ai-data-analytics-project-roadmap/customs_duties_trend_selected_countries.csv")
    shipments = pd.read_csv("ai-data-analytics-project-roadmap/synthetic_customs_shipments.csv")
    return trend, shipments

trend, shipments = load_data()

st.subheader("1) World Bank – Customs Import Duties (LCU)")

# Build options safely and choose only defaults that exist
options = sorted(trend["Country Name"].dropna().unique().tolist())
preferred_defaults = ["Nigeria", "United Kingdom", "United States", "China", "South Africa"]
default = [c for c in preferred_defaults if c in options]
if not default:
    default = options[:2] if len(options) >= 2 else options

cc = st.multiselect("Select countries", options, default=default)

plot_df = trend[trend["Country Name"].isin(cc)]
if plot_df.empty:
    st.info("No data available for the selected countries.")
else:
    pivot = plot_df.pivot_table(index="Year", columns="Country Name", values="Customs_Import_Duties_LCU", aggfunc="mean")
    # Ensure Year is sorted numerically if read as string
    pivot.index = pd.to_numeric(pivot.index, errors="coerce")
    pivot = pivot.sort_index()
    st.line_chart(pivot)

st.subheader("2) Synthetic Shipment Risk Explorer")
col1, col2, col3 = st.columns(3)
with col1:
    dest = st.selectbox("Destination", ["All"] + sorted(shipments["destination_country"].dropna().unique().tolist()))
with col2:
    mode = st.selectbox("Transport Mode", ["All"] + sorted(shipments["transport_mode"].dropna().unique().tolist()))
with col3:
    min_score = st.slider("Min Risk Score", 0, 100, 60)

filt = shipments.copy()
if dest != "All":
    filt = filt[filt["destination_country"] == dest]
if mode != "All":
    filt = filt[filt["transport_mode"] == mode]
filt = filt[filt["risk_score"] >= min_score]

st.write(f"Showing {len(filt):,} shipments (risk_score ≥ {min_score})")

# Histogram of risk scores
if not filt.empty:
    fig = plt.figure()
    plt.hist(filt["risk_score"], bins=30)
    plt.xlabel("Risk Score")
    plt.ylabel("Count")
    plt.title("Distribution of Risk Scores")
    st.pyplot(fig)
else:
    st.info("No shipments match the current filters.")

st.dataframe(filt.head(50))

st.download_button("⬇️ Download filtered shipments (CSV)", data=filt.to_csv(index=False), file_name="filtered_shipments.csv", mime="text/csv")
