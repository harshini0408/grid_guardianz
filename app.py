# app_noml.py – NBCFDC Dashboard (No ML version)
import os
import logging
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------
# Basic setup
# ---------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nbcfdc-app-noml")

st.set_page_config(page_title="NBCFDC Lending Dashboard", layout="wide")
st.title("🏦 NBCFDC Credit Scoring & Direct Digital Lending (Demo – No ML)")

st.markdown("""
Welcome to the simplified **non‑ML version** of the NBCFDC Beneficiary Scoring and Lending Dashboard.
All UI, tabs, and logic remain exactly as before — only the ML model predictions are replaced by static placeholder logic.
""")

# ---------------------------------
# Placeholder logic for scoring and risk determination
# ---------------------------------
def mock_score_logic(df_row):
    """Simulated scoring mechanism – returns pseudo scores for demo."""
    amount = df_row.get("Loan_Amount", 0)
    repayment = df_row.get("On_Time_Repayment_Rate", 0.8)
    utilization = df_row.get("Loan_Utilization_Pct", 50)
    score = min(100, max(20, 80 + repayment * 15 - utilization * 0.05 - amount/100000))
    if score >= 75:
        risk = "Low Risk - High Need"
    elif score >= 55:
        risk = "Medium Risk"
    else:
        risk = "High Risk"
    decision = "AUTO-APPROVE" if score >= 70 else ("MANUAL REVIEW" if score >= 50 else "REJECT")
    return round(score, 2), risk, decision

# ---------------------------------
# Tabs
# ---------------------------------
tabs = st.tabs(["Beneficiary", "Loan Officer / Channel Partner", "Admin"])

# ---------------------------------
# Beneficiary Portal
# ---------------------------------
with tabs[0]:
    st.header("👤 Beneficiary Portal — Check your eligibility")

    col1, col2 = st.columns([1,1])
    with col1:
        Loan_Amount = st.number_input("Loan Amount (₹)", min_value=0.0, value=25000.0, step=500.0)
        Loan_Tenure_Months = st.number_input("Loan Tenure (Months)", min_value=1, max_value=120, value=12)
        Loan_Type = st.selectbox("Loan Type", ["Business","Education","Personal","Agriculture"])
        On_Time_Repayment_Rate = st.slider("On-Time Repayment Rate (0-1)", 0.0, 1.0, 0.90)
        Delinquencies = st.number_input("Delinquencies (count)", min_value=0, value=0)
        Loan_Utilization_Pct = st.slider("Loan Utilization (%)", 0, 100, 70)
        Repeat_Borrower = st.selectbox("Repeat Borrower", ["Yes","No"])

    with col2:
        Electricity_Consumption_kWh = st.number_input("Electricity Consumption (kWh/month)", min_value=0.0, value=200.0)
        Mobile_Recharge_Freq = st.number_input("Mobile Recharge Frequency (per month)", min_value=0, value=6)
        Mobile_Recharge_Amount = st.number_input("Avg Mobile Recharge Amount (₹)", min_value=0.0, value=150.0)
        Utility_Bill_Regular = st.selectbox("Utility Bill Regular", ["Yes","No"])
        Caste_Category = st.selectbox("Caste Category", ["SC","ST","OBC","General"])
        Region = st.selectbox("Region", ["Urban","Rural","Semi-Urban"])
        Education_Level = st.selectbox("Education Level", ["No Formal Education","Primary","Secondary","Graduate","Post-Graduate"])

    if st.button("Check Eligibility"):
        data = {
            "Loan_Amount": Loan_Amount,
            "Loan_Tenure_Months": Loan_Tenure_Months,
            "Loan_Type": Loan_Type,
            "On_Time_Repayment_Rate": On_Time_Repayment_Rate,
            "Delinquencies": Delinquencies,
            "Loan_Utilization_Pct": Loan_Utilization_Pct,
            "Repeat_Borrower": Repeat_Borrower,
            "Electricity_Consumption_kWh": Electricity_Consumption_kWh,
            "Mobile_Recharge_Freq": Mobile_Recharge_Freq,
            "Mobile_Recharge_Amount": Mobile_Recharge_Amount,
            "Utility_Bill_Regular": Utility_Bill_Regular,
            "Caste_Category": Caste_Category,
            "Region": Region,
            "Education_Level": Education_Level
        }
        df_row = pd.Series(data)
        score, risk, decision = mock_score_logic(df_row)

        st.subheader(f"Composite Score: {score}")
        st.write(f"**Risk Band:** {risk}")
        st.write(f"**Decision:** {decision}")

        # Visual representation
        gauge_color = "green" if decision == "AUTO-APPROVE" else ("orange" if decision == "MANUAL REVIEW" else "red")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Composite Score"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': gauge_color}}
        ))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# Loan Officer / Channel Partner tab
# ---------------------------------
with tabs[1]:
    st.header("🏦 Loan Officer / Channel Partner")
    st.markdown("Upload CSV file (with same structure as beneficiary inputs) for **batch scoring**.")

    uploaded = st.file_uploader("Upload Beneficiary CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Run Batch Evaluation"):
            results = []
            for _, row in df.iterrows():
                s, r, d = mock_score_logic(row)
                results.append((s, r, d))
            df["Composite_Score"], df["Risk_Band"], df["Decision"] = zip(*results)
            st.success("Batch processed successfully!")
            st.write(df.head())
            fig = px.histogram(df, x="Composite_Score", nbins=30, color="Risk_Band")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download Results CSV", df.to_csv(index=False).encode('utf-8'), "batch_results.csv")

# ---------------------------------
# Admin Dashboard tab
# ---------------------------------
with tabs[2]:
    st.header("📊 Admin Dashboard")
    st.markdown("Upload a previously scored CSV to review analytics and fairness insights.")
    admin_file = st.file_uploader("Upload Scored CSV", type=["csv"], key="admin_upload")

    if admin_file:
        admin_df = pd.read_csv(admin_file)
        st.write("### Summary Overview")
        st.metric("Total Records", len(admin_df))
        st.metric("Average Score", f"{admin_df['Composite_Score'].mean():.2f}" if 'Composite_Score' in admin_df else "N/A")

        if 'Risk_Band' in admin_df.columns:
            counts = admin_df['Risk_Band'].value_counts()
            st.bar_chart(counts)

        if 'Region' in admin_df.columns:
            fig_region = px.histogram(admin_df, x="Region", color="Risk_Band", barmode="group")
            st.plotly_chart(fig_region, use_container_width=True)

        if 'Caste_Category' in admin_df.columns and 'Composite_Score' in admin_df.columns:
            st.subheader("Fairness Check: Caste-wise Average Score")
            caste_mean = admin_df.groupby('Caste_Category')['Composite_Score'].mean().reset_index()
            st.dataframe(caste_mean)

        st.download_button("Download Admin Snapshot", admin_df.to_csv(index=False).encode('utf-8'),
                           "admin_snapshot.csv")

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.caption("Demo version without ML for NBCFDC dashboard. All model predictions replaced with rule-based scoring.")
