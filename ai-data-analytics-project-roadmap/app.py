import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Customs Risk Analytics", layout="wide")

st.title("📦 Customs Cargo Risk Analytics – Demo (with Auto‑Summary)")

@st.cache_data
def load_data():
    trend = pd.read_csv("ai-data-analytics-project-roadmap/customs_duties_trend_selected_countries.csv")
    shipments = pd.read_csv("ai-data-analytics-project-roadmap/synthetic_customs_shipments.csv")
    return trend, shipments

trend, shipments = load_data()

# -----------------------------
# Helpers
# -----------------------------

def pct(n):
    try:
        return f"{n:,.2f}%"
    except Exception:
        return "–"


def fmt(n):
    try:
        if pd.isna(n):
            return "–"
        if isinstance(n, (int, float)):
            return f"{n:,.2f}"
        return str(n)
    except Exception:
        return "–"


def summarize_trend(trend_df: pd.DataFrame, selected_countries: list[str]) -> str:
    if trend_df.empty or not selected_countries:
        return "No trend selection made."

    # keep only selected
    t = trend_df[trend_df["Country Name"].isin(selected_countries)].copy()
    # ensure numeric year
    t["Year"] = pd.to_numeric(t["Year"], errors="coerce")
    t = t.dropna(subset=["Year"])  # remove bad years

    lines = ["**World Bank – Customs Import Duties (LCU) summary:**"]

    # For each country, find latest two non-null observations and compute % change
    metric_col = "Customs_Import_Duties_LCU"
    for c in selected_countries:
        tc = t[t["Country Name"] == c].dropna(subset=[metric_col])
        if tc.empty:
            lines.append(f"- {c}: no data available.")
            continue
        tc = tc.sort_values("Year")
        latest_row = tc.iloc[-1]
        latest_year = int(latest_row["Year"]) if pd.notna(latest_row["Year"]) else None
        latest_val = latest_row[metric_col]
        prev_row = tc.iloc[-2] if len(tc) >= 2 else None
        if prev_row is not None:
            prev_val = prev_row[metric_col]
            change = ((latest_val - prev_val) / prev_val) * 100 if prev_val not in (0, None, np.nan) else np.nan
            ch_txt = pct(change) if pd.notna(change) else "–"
            lines.append(
                f"- {c}: latest **{fmt(latest_val)}** in **{latest_year}**, vs {fmt(prev_val)} in {int(prev_row['Year'])} (Δ {ch_txt})."
            )
        else:
            lines.append(f"- {c}: latest **{fmt(latest_val)}** in **{latest_year}** (no prior year to compare).")

    # Aggregate latest year across countries
    latest_year_overall = t.groupby("Country Name")["Year"].max().dropna()
    if not latest_year_overall.empty:
        ly = int(latest_year_overall.median())
        lines.append(f"- Median latest observation year across selections: **{ly}**.")

    return "\n".join(lines)


def summarize_shipments(df: pd.DataFrame) -> str:
    if df.empty:
        return "No shipments match the current filters. Adjust filters to see a summary."

    total = len(df)
    avg_risk = df["risk_score"].mean()
    p80 = (df["risk_score"] >= 80).mean() * 100
    top_modes = df["transport_mode"].value_counts().head(3)
    top_dest = df["destination_country"].value_counts().head(3)
    top_origin = df["origin_country"].value_counts().head(3) if "origin_country" in df.columns else pd.Series(dtype=int)
    top_hs = df["hs_code"].value_counts().head(5) if "hs_code" in df.columns else pd.Series(dtype=int)

    # common lanes (origin→dest) if columns exist
    lanes = None
    if all(col in df.columns for col in ["origin_country", "destination_country"]):
        lanes = (
            df.assign(lane=df["origin_country"] + " → " + df["destination_country"])
            ["lane"].value_counts().head(5)
        )

    parts = [
        f"**Filtered shipment summary:** Showing **{total:,}** shipments. Avg risk **{fmt(avg_risk)}**; High‑risk (≥80) **{fmt(p80)}%**.",
        "- Top transport modes: "
        + ", ".join([f"{k} ({v:,})" for k, v in top_modes.items()]) if not top_modes.empty else "- Top transport modes: –",
        "- Top destinations: "
        + ", ".join([f"{k} ({v:,})" for k, v in top_dest.items()]) if not top_dest.empty else "- Top destinations: –",
    ]

    if not top_origin.empty:
        parts.append("- Top origins: " + ", ".join([f"{k} ({v:,})" for k, v in top_origin.items()]))
    if not top_hs.empty:
        parts.append("- Frequent HS codes: " + ", ".join([f"{k} ({v:,})" for k, v in top_hs.items()]))
    if lanes is not None and not lanes.empty:
        parts.append("- Common lanes: " + ", ".join([f"{k} ({v:,})" for k, v in lanes.items()]))

    # risk buckets
    buckets = pd.cut(
        df["risk_score"],
        bins=[0, 40, 60, 80, 100],
        labels=["Low (0–39)", "Moderate (40–59)", "Elevated (60–79)", "High (80–100)"]
    ).value_counts().reindex(["Low (0–39)", "Moderate (40–59)", "Elevated (60–79)", "High (80–100)"], fill_value=0)
    parts.append("- Risk buckets: " + ", ".join([f"{k}: {v:,}" for k, v in buckets.items()]))

    return "\n".join(parts)


def build_overall_summary(trend_df: pd.DataFrame, countries: list[str], filtered_shipments: pd.DataFrame) -> str:
    trend_txt = summarize_trend(trend_df, countries)
    ship_txt = summarize_shipments(filtered_shipments)
    return (
        "### 🧾 Executive Summary\n"
        + trend_txt
        + "\n\n"
        + ship_txt
    )

# -----------------------------
# UI – Section 1: World Bank trend
# -----------------------------

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
    pivot = pd.DataFrame()
else:
    pivot = plot_df.pivot_table(
        index="Year",
        columns="Country Name",
        values="Customs_Import_Duties_LCU",
        aggfunc="mean",
    )
    # Ensure Year is sorted numerically if read as string
    pivot.index = pd.to_numeric(pivot.index, errors="coerce")
    pivot = pivot.sort_index()
    st.line_chart(pivot)

# -----------------------------
# UI – Section 2: Shipment Risk Explorer
# -----------------------------

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

st.download_button(
    "⬇️ Download filtered shipments (CSV)",
    data=filt.to_csv(index=False),
    file_name="filtered_shipments.csv",
    mime="text/csv",
)

# -----------------------------
# Auto‑generated Summary Section
# -----------------------------

summary_md = build_overall_summary(trend, cc, filt)

with st.expander("🧾 Executive Summary", expanded=True):
    st.markdown(summary_md)

# Optional: show raw numbers used in the summary
with st.expander("Details behind the summary"):
    c1, c2 = st.columns(2)
    with c1:
        if not pivot.empty:
            st.write("Latest values by country (last year available):")
            latest_vals = (
                plot_df.dropna(subset=["Customs_Import_Duties_LCU"])\
                .sort_values(["Country Name", "Year"])\
                .groupby("Country Name").tail(1)\
                [["Country Name", "Year", "Customs_Import_Duties_LCU"]]
            )
            st.dataframe(latest_vals.reset_index(drop=True))
        else:
            st.caption("No trend data for the selected countries.")
    with c2:
        if not filt.empty:
            st.write("Top HS codes (filtered):")
            if "hs_code" in filt.columns:
                st.dataframe(filt["hs_code"].value_counts().head(10).rename_axis("HS Code").reset_index(name="Count"))
            else:
                st.caption("No HS code column present in the dataset.")
        else:
            st.caption("No filtered shipments to summarize.")
