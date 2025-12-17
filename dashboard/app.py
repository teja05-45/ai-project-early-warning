import sys
import os

# ---------------- PATH FIX ----------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ---------------- IMPORTS ----------------
import streamlit as st
import pandas as pd
import joblib
import shap

from src.simulation import WhatIfSimulator
from src.feature_engineering import engineer_features
from src.explainability import (
    get_top_risk_drivers,
    compare_driver_change
)

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Delivery Risk Early Warning System",
    layout="wide"
)

# ---------------- LOADERS ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/raw/project_delivery_risk.csv")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/delivery_risk_model.joblib")
    feature_cols = joblib.load("models/feature_columns.joblib")
    explainer = shap.TreeExplainer(model)
    return model, feature_cols, explainer

df = load_data()
model, feature_cols, explainer = load_artifacts()
simulator = WhatIfSimulator()

# ---------------- HELPERS ----------------
def risk_bucket(score):
    if score < 30:
        return "🟢 Low Risk"
    elif score < 60:
        return "🟡 Medium Risk"
    elif score < 80:
        return "🟠 High Risk"
    else:
        return "🔴 Critical Risk"

def risk_direction(risks, threshold=10):
    if len(risks) < 2:
        return "➖ No Trend Data", 0
    change = risks[-1] - risks[-2]
    if change > threshold:
        return "🔺 Risk Increasing (Early Warning)", change
    elif change < -threshold:
        return "🔻 Risk Improving", change
    else:
        return "➖ Risk Stable", change

def recommend_actions(drivers):
    actions = []
    for feature, _ in drivers:
        f = feature.lower()
        if "scope" in f:
            actions.append("📉 Reduce scope volatility and freeze backlog")
        elif "attrition" in f:
            actions.append("👥 Address burnout or add backup resources")
        elif "blocker" in f or "dependency" in f:
            actions.append("⛔ Resolve external dependencies early")
        elif "bug" in f or "quality" in f:
            actions.append("🧪 Improve testing & defect prevention")
        elif "work" in f or "hours" in f:
            actions.append("⏱️ Reduce overtime to stabilize productivity")
    return list(set(actions))[:3]

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Executive Summary",
        "📈 Project View",
        "📊 Risk Trends",
        "🔍 Risk Explanation",
        "🔄 What-If Simulator"
    ]
)

# =====================================================
# 🏠 EXECUTIVE SUMMARY
# =====================================================
if page == "🏠 Executive Summary":
    st.title("🏠 Executive Delivery Risk Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Projects", df["project_id"].nunique())
    col2.metric("High Risk Sprints", int((df["delivery_risk"] == 1).sum()))
    col3.metric("Avg Risk (%)", f"{df['delivery_risk'].mean()*100:.1f}")

    summary = []
    for pid in df["project_id"].unique():
        last = df[df["project_id"] == pid].sort_values("sprint_id").iloc[-1]
        risk = simulator.predict_risk(last) * 100
        summary.append({
            "Project": pid,
            "Current Risk (%)": round(risk, 1),
            "Risk Level": risk_bucket(risk),
            "Milestone Slippage (days)": int(last["milestone_slippage_days"])
        })

    st.subheader("🚨 High Risk Projects")
    st.dataframe(
        pd.DataFrame(summary)
        .sort_values("Current Risk (%)", ascending=False)
        .head(10)
    )

# =====================================================
# 📈 PROJECT VIEW
# =====================================================
elif page == "📈 Project View":
    st.title("📈 Project Deep Dive")

    pid = st.selectbox("Select Project", df["project_id"].unique())
    sid = st.selectbox(
        "Select Sprint",
        df[df["project_id"] == pid]["sprint_id"]
    )

    row = df[
        (df["project_id"] == pid) &
        (df["sprint_id"] == sid)
    ].iloc[0]

    risk = simulator.predict_risk(row) * 100
    st.metric("Delivery Risk Score", f"{risk:.1f}%", risk_bucket(risk))
    st.json(row.to_dict())

# =====================================================
# 📊 RISK TRENDS
# =====================================================
elif page == "📊 Risk Trends":
    st.title("📊 Risk Trends & Early Warnings")

    pid = st.selectbox("Select Project", df["project_id"].unique())
    pdf = df[df["project_id"] == pid].sort_values("sprint_id")

    pdf["predicted_risk"] = pdf.apply(
        lambda r: simulator.predict_risk(r) * 100, axis=1
    )

    trend, delta = risk_direction(pdf["predicted_risk"].tolist())

    st.subheader("🚨 Risk Direction")
    st.write(f"**{trend}** (Change: {delta:.1f}%)")

    st.subheader("📈 Risk Over Sprints")
    st.line_chart(pdf.set_index("sprint_id")["predicted_risk"])

    st.dataframe(
        pdf[
            [
                "sprint_id",
                "predicted_risk",
                "velocity_change_pct",
                "scope_change_requests",
                "milestone_slippage_days"
            ]
        ]
    )

# =====================================================
# 🔍 RISK EXPLANATION
# =====================================================
elif page == "🔍 Risk Explanation":
    st.title("🔍 Why Is Delivery Risk High?")

    pid = st.selectbox("Select Project", df["project_id"].unique())
    sid = st.selectbox(
        "Select Sprint",
        df[df["project_id"] == pid]["sprint_id"]
    )

    project_df = df[df["project_id"] == pid].sort_values("sprint_id")

    row = project_df[project_df["sprint_id"] == sid].iloc[0]
    X_row = engineer_features(pd.DataFrame([row]))[feature_cols]

    shap_current = explainer.shap_values(X_row)
    drivers = get_top_risk_drivers(shap_current, feature_cols)

    st.subheader("🚨 Top Risk Drivers")
    for f, v in drivers:
        direction = "↑ Increasing Risk" if v > 0 else "↓ Reducing Risk"
        st.write(f"**{f}** — {direction} ({v:.2f})")

    # ---- Driver change vs last sprint ----
    prev = project_df[project_df["sprint_id"] < sid]
    if not prev.empty:
        prev_row = prev.iloc[-1]
        X_prev = engineer_features(pd.DataFrame([prev_row]))[feature_cols]
        shap_prev = explainer.shap_values(X_prev)

        changes = compare_driver_change(
            shap_current,
            shap_prev,
            feature_cols
        )

        st.subheader("📈 What Changed Since Last Sprint")
        for f, d in changes:
            arrow = "⬆️" if d > 0 else "⬇️"
            st.write(f"**{f}** {arrow} ({d:.2f})")

    st.subheader("🛠️ Recommended Actions")
    actions = recommend_actions(drivers)
    if actions:
        for a in actions:
            st.write(f"- {a}")
    else:
        st.write("✅ No immediate intervention required")

# =====================================================
# 🔄 WHAT-IF SIMULATOR
# =====================================================
elif page == "🔄 What-If Simulator":
    st.title("🔄 What-If Simulation")

    row = df.iloc[0]

    scope = st.slider("Reduce Scope (%)", -50, 0, -20)
    blockers = st.slider("Reduce Blockers (%)", -80, 0, -30)
    hours = st.slider("Reduce Work Hours (%)", -40, 0, -20)

    result = simulator.simulate(
        row,
        {
            "planned_story_points": scope / 100,
            "dependency_blockers": blockers / 100,
            "avg_work_hours": hours / 100
        }
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Risk", f"{result['baseline_risk']}%")
    col2.metric("New Risk", f"{result['new_risk']}%")
    col3.metric("Risk Change", f"{result['risk_change']}%")
