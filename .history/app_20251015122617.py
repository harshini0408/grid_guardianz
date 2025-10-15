# app.py - NBCFDC Super Dashboard (corrected)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import os

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(page_title="NBCFDC Credit Scoring (Super)", layout="wide", initial_sidebar_state="auto")
st.title("🏦 NBCFDC Credit Scoring & Direct Digital Lending (Super)")

st.markdown(
    "Multi-role demo: *Beneficiary* | *Loan Officer* | *Admin* — "
    "Includes single/batch prediction, SHAP explainability (bar + waterfall fallback), and auto-decision suggestions."
)

# ------------------------------
# Load artifacts (cached)
# ------------------------------
@st.cache_resource
def load_artifacts():
    reg = joblib.load("regression_model.pkl")
    clf = joblib.load("classification_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")  # expected dict {col: LabelEncoder}
    return reg, clf, scaler, encoders

try:
    reg_model, clf_model, scaler, encoders = load_artifacts()
except Exception as e:
    st.error("Error loading model artifacts. Make sure regression_model.pkl, classification_model.pkl, scaler.pkl and encoders.pkl exist in this folder.")
    st.exception(e)
    st.stop()

# ------------------------------
# Feature ordering (must match training)
# ------------------------------
FEATURE_ORDER = [
    "Loan_Amount","Loan_Tenure_Months","Loan_Type","On_Time_Repayment_Rate",
    "Delinquencies","Loan_Utilization_Pct","Repeat_Borrower",
    "Electricity_Consumption_kWh","Mobile_Recharge_Freq","Mobile_Recharge_Amount",
    "Utility_Bill_Regular","Caste_Category","Region","Education_Level"
]

# ------------------------------
# Helpers: encoding, preprocessing
# ------------------------------
def safe_encode_column(series, le):
    """Encode a pandas Series with a saved LabelEncoder le. Unseen -> index 0 fallback."""
    out = []
    classes = list(le.classes_)
    for v in series.astype(str).tolist():
        if v in classes:
            out.append(int(np.where(le.classes_ == v)[0][0]))
        else:
            # fallback to the most frequent/first class index (0) to avoid transform error
            out.append(0)
    return out

def preprocess_df(df):
    """
    Input: raw DataFrame (may be single-row or batch)
    Output: (X_scaled, feature_names) where X_scaled is a numpy array ready for model.predict
    """
    X = df.copy()

    # Ensure expected columns exist; create defaults for missing
    for c in FEATURE_ORDER:
        if c not in X.columns:
            if c in ["Loan_Amount","Loan_Tenure_Months","On_Time_Repayment_Rate",
                     "Delinquencies","Loan_Utilization_Pct","Electricity_Consumption_kWh",
                     "Mobile_Recharge_Freq","Mobile_Recharge_Amount"]:
                X[c] = 0
            else:
                X[c] = "Unknown"

    # Encode categorical columns using saved encoders
    for col, le in encoders.items():
        if col in X.columns:
            encoded = safe_encode_column(X[col], le)
            X[col] = encoded

    # Yes/No -> 1/0 for specific columns if still strings
    for bcol in ["Repeat_Borrower", "Utility_Bill_Regular"]:
        if bcol in X.columns:
            X[bcol] = X[bcol].map({'Yes':1, 'No':0}).fillna(0).astype(int)

    # Keep only FEATURE_ORDER and cast to float
    X_final = X[FEATURE_ORDER].astype(float)

    # Scale and return
    X_scaled = scaler.transform(X_final)
    return X_scaled, list(X_final.columns)

# ------------------------------
# Risk helpers
# ------------------------------
def risk_color(risk_band):
    rb = str(risk_band).lower()
    if ("low risk" in rb and "high need" in rb) or ("low risk" in rb and "low need" in rb) or ("low" in rb and "risk" in rb):
        return "green"
    if "medium" in rb or ("low" in rb and "need" in rb and "low" not in rb):
        return "orange"
    if "high" in rb:
        return "red"
    return "gray"

def auto_decision(score, risk_band):
    rb = str(risk_band).lower()
    if score >= 70 and "low" in rb:
        return "AUTO-APPROVE"
    if 50 <= score < 70:
        return "MANUAL REVIEW"
    return "REJECT / DEEP REVIEW"

# ------------------------------
# SHAP helpers
# ------------------------------
def shap_bar_plot(shap_values, feature_names, max_display=10):
    # shap_values: Explanation object from explainer(Xs)
    try:
        vals = shap_values.values[0] if hasattr(shap_values, 'values') else np.array(shap_values)[0]
    except Exception:
        vals = np.array(shap_values)[0]
    df = pd.DataFrame({'feature': feature_names, 'shap': vals})
    df['abs_shap'] = df['shap'].abs()
    df = df.sort_values('abs_shap', ascending=False).head(max_display)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.5 * len(df))))
    ax.barh(df['feature'][::-1], df['shap'][::-1])
    ax.set_xlabel("SHAP value")
    ax.set_title("Top feature contributions (SHAP)")
    plt.tight_layout()
    return fig, df

def shap_waterfall_legacy_plot(explainer, shap_values, feature_names):
    # Attempt legacy waterfall; if fails, raise exception
    try:
        # shap's internal waterfall function (may differ by version)
        # Many shap versions provide this; if not, this will raise.
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values.values[0],
                                              feature_names=feature_names, show=False)
        fig = plt.gcf()
        return fig
    except Exception as e:
        raise e

# ------------------------------
# UI Tabs
# ------------------------------
tabs = st.tabs(["Beneficiary", "Loan Officer / Channel Partner", "Admin"])

# ------------------------------
# Beneficiary tab
# ------------------------------
with tabs[0]:
    st.header("👤 Beneficiary Portal — Check Your Score")
    st.markdown("Enter your details and click *Check*. You'll receive a composite score, risk band, top SHAP contributors, and a recommendation.")

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

    if st.button("Check"):
        single = pd.DataFrame([{
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
        }])

        # Preprocess and predict
        try:
            Xs, feat_names = preprocess_df(single)
            score = float(reg_model.predict(Xs)[0])
            risk = str(clf_model.predict(Xs)[0])
        except Exception as e:
            st.error("Prediction failed during preprocessing or model inference.")
            st.exception(e)
            st.stop()

        color = risk_color(risk)
        decision = auto_decision(score, risk)

        st.markdown(f"### Composite Credit Score: *{score:.2f}*")
        # risk badge
        st.markdown(f"<div style='display:flex;align-items:center;gap:12px'>"
                    f"<div style='width:16px;height:16px;background:{color};border-radius:4px;'></div>"
                    f"<div style='font-size:16px'><b>Risk Band:</b> {risk}</div>"
                    f"</div>", unsafe_allow_html=True)

        # decision badge
        decision_color = "green" if decision=="AUTO-APPROVE" else ("orange" if decision=="MANUAL REVIEW" else "red")
        st.markdown(f"<div style='margin-top:8px;padding:8px;border-radius:8px;background:#f6f6f6;'>"
                    f"<b>Recommendation:</b> <span style='color:{decision_color};font-weight:700'>{decision}</span>"
                    f"</div>", unsafe_allow_html=True)

        # simple bar + gauge (gauge preferred)
        # Bar (fallback simple)
        fig_bar = px.bar(x=["Composite Score"], y=[score], text=[f"{score:.2f}"], range_y=[0,100],
                         labels={'x':'Metric','y':'Score'}, title="Composite Credit Score (bar)")
        fig_bar.update_traces(marker_color=color, textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

        # Gauge-style indicator
        if score >= 80:
            gauge_color = "green"
            zone = "Low Risk"
        elif 60 <= score < 80:
            gauge_color = "orange"
            zone = "Medium Risk"
        else:
            gauge_color = "red"
            zone = "High Risk"

        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            number={'suffix': " /100", 'font': {'size': 36}},
            delta={'reference': 70, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 60], 'color': 'rgba(255,0,0,0.3)'},
                    {'range': [60, 80], 'color': 'rgba(255,165,0,0.3)'},
                    {'range': [80, 100], 'color': 'rgba(0,128,0,0.3)'}
                ],
                'threshold': {
                    'line': {'color': gauge_color, 'width': 6},
                    'thickness': 0.75,
                    'value': score
                }
            },
            title={'text': f"<b>Composite Credit Score</b><br><span style='font-size:0.8em;color:gray'>{zone}</span>"}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        # SHAP explainability
        st.subheader("Top feature contributions (SHAP)")
        try:
            explainer = shap.Explainer(reg_model, Xs)  # create explainer with background Xs
            shap_vals = explainer(Xs)

            fig_shap, df_shap = shap_bar_plot(shap_vals, feat_names, max_display=10)
            st.pyplot(fig_shap)
            st.table(df_shap[['feature','shap']].head(5).assign(SHAP_Impact=lambda d: d['shap'].round(4)))

            # attempt waterfall; fallback if not available
            try:
                fig_wf = shap_waterfall_legacy_plot(explainer, shap_vals, feat_names)
                if fig_wf is not None:
                    st.subheader("SHAP Waterfall (detailed contribution)")
                    st.pyplot(fig_wf)
            except Exception:
                st.info("Detailed SHAP waterfall not available for this shap version; using bar + table.")
        except Exception as e:
            st.warning("SHAP explanation failed. Showing only prediction.")
            st.exception(e)

        # Next actions
        if decision == "AUTO-APPROVE":
            st.success("This application qualifies for automatic sanction. Next step: generate sanction letter and disburse.")
        elif decision == "MANUAL REVIEW":
            st.info("This application requires manual review by Loan Officer.")
        
# ------------------------------
# Loan Officer / Channel Partner tab
# ------------------------------
with tabs[1]:
    st.header("🏦 Loan Officer / Channel Partner")
    st.markdown("Upload CSV with beneficiary rows for batch prediction. Required columns: same as UI fields (Loan_Amount, Loan_Tenure_Months, Loan_Type, ...).")

    uploaded = st.file_uploader("Upload CSV for batch scoring", type=["csv"], key="batch_upload")
    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error("Failed to read CSV")
            st.exception(e)
            batch_df = None

        if batch_df is not None:
            st.dataframe(batch_df.head(5))
            if st.button("Run Batch Prediction"):
                try:
                    Xs_batch, feat_names_batch = preprocess_df(batch_df)
                    scores = reg_model.predict(Xs_batch)
                    bands = clf_model.predict(Xs_batch)
                    batch_df['Composite_Credit_Score'] = np.round(scores, 2)
                    batch_df['Risk_Band'] = bands
                    # Recommendation per row using predicted risk band
                    batch_df['Recommendation'] = [auto_decision(s, b) for s, b in zip(batch_df['Composite_Credit_Score'], batch_df['Risk_Band'])]
                    st.success(f"Predicted {len(batch_df)} rows.")
                    st.dataframe(batch_df)

                    # Histogram
                    fig_hist = px.histogram(batch_df, x="Composite_Credit_Score", nbins=30, color="Risk_Band")
                    st.plotly_chart(fig_hist, use_container_width=True)

                    # Download
                    csv_bytes = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download scored CSV", data=csv_bytes, file_name="batch_scored.csv", mime="text/csv")
                except Exception as e:
                    st.error("Batch prediction failed.")
                    st.exception(e)

# ------------------------------
# Admin tab
# ------------------------------
with tabs[2]:
    st.header("📊 Admin Dashboard")
    st.markdown("Upload a scored CSV (with Composite_Credit_Score and Risk_Band) to produce analytics & fairness checks.")

    admin_upload = st.file_uploader("Admin upload (scored CSV)", type=["csv"], key="admin_upload")
    if admin_upload:
        try:
            admin_df = pd.read_csv(admin_upload)
        except Exception as e:
            st.error("Unable to read admin CSV")
            st.exception(e)
            admin_df = None

        if admin_df is not None:
            st.markdown("### Portfolio Summary")
            total = len(admin_df)
            avg_score = admin_df['Composite_Credit_Score'].mean() if 'Composite_Credit_Score' in admin_df.columns else np.nan
            st.metric("Total records", total)
            st.metric("Average Composite Score", f"{avg_score:.2f}" if not np.isnan(avg_score) else "N/A")

            if 'Risk_Band' in admin_df.columns:
                counts = admin_df['Risk_Band'].value_counts().to_dict()
                st.write("Risk Band counts:", counts)

            # Score distribution
            if 'Composite_Credit_Score' in admin_df.columns:
                fig = px.histogram(admin_df, x="Composite_Credit_Score", nbins=30, color="Risk_Band" if 'Risk_Band' in admin_df.columns else None)
                st.plotly_chart(fig, use_container_width=True)

            # Region vs risk
            if 'Region' in admin_df.columns:
                fig2 = px.histogram(admin_df, x="Region", color="Risk_Band" if 'Risk_Band' in admin_df.columns else None, barmode="group")
                st.plotly_chart(fig2, use_container_width=True)

            # Fairness quick checks (Caste)
            if 'Caste_Category' in admin_df.columns and 'Composite_Credit_Score' in admin_df.columns:
                st.subheader("Fairness: Caste-wise average score")
                caste_mean = admin_df.groupby('Caste_Category')['Composite_Credit_Score'].mean().reset_index()
                st.table(caste_mean)

            # Download processed analytics snapshot
            csvb = admin_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download analytics snapshot", data=csvb, file_name="admin_snapshot.csv", mime="text/csv")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("*Notes:* The 'auto decision' logic is rule-based for demo. For production use implement authentication, audit logs, and secure PII storage.")