import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from app.utils.drift import build_drift_report, get_drift_summary


def render_evidently_section(train_df, test_df, feature_meta):
    st.subheader("📡 Evidently AI — Data Drift Monitoring")

    st.info(
        "**Simulation note:** In production, reference data = training set "
        "and current data = live incoming orders. Here we simulate using "
        "the test set as current data."
    )

    if st.button("🔄 Run Drift Analysis", type="primary"):
        with st.spinner("Running Evidently drift analysis..."):
            features = feature_meta["final_features"]
            report   = build_drift_report(train_df, test_df, features)
            summary  = get_drift_summary(report)

        # ── Drift summary metrics ────────────────────────────────
        st.divider()
        m1, m2, m3, m4 = st.columns(4)

        drift_detected = summary["dataset_drift_detected"]
        with m1:
            if drift_detected:
                st.error("🔴 Drift Detected")
            else:
                st.success("🟢 No Drift Detected")

        with m2:
            st.metric(
                "Drifted Features",
                f"{summary['n_drifted_features']} / {summary['n_features_checked']}"
            )
        with m3:
            st.metric(
                "Share Drifted",
                f"{summary['share_drifted_features']*100:.1f}%"
            )
        with m4:
            st.metric("Features Checked", summary["n_features_checked"])

        # ── Drifted features table ───────────────────────────────
        if summary["drifted_features"]:
            st.markdown("**Drifted Features Detail:**")
            drift_df = pd.DataFrame(summary["drifted_features"])
            drift_df = drift_df.sort_values("p_value")
            st.dataframe(drift_df, use_container_width=True)
        else:
            st.success("✅ No individual features showed significant drift.")

        # ── Full Evidently HTML report ───────────────────────────
        st.markdown("**Full Evidently Report:**")
        report_html = report.get_html()
        st.components.v1.html(report_html, height=600, scrolling=True)


def render(model, feature_meta, validation_results, shap_importance,
           train_df, test_df):
    st.header("🛠️ Developer Dashboard")
    st.markdown(
        "Internal monitoring view — data drift detection."
    )

    render_evidently_section(train_df, test_df, feature_meta)