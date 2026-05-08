import pandas as pd
import numpy as np
# NEW
from evidently import Report
from evidently.presets import DataDriftPreset, DataQualityPreset

from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
)


NUMERICAL_FEATURES = [
    "estimated_delivery_days", "total_freight", "total_price",
    "product_weight_g", "payment_installments", "item_count",
    "seller_avg_variance", "seller_late_rate", "seller_order_count",
    "customer_state_avg_variance", "seller_state_avg_variance",
    "category_avg_variance", "freight_price_ratio",
]

CATEGORICAL_FEATURES = [
    "same_state", "same_region", "purchase_month",
    "purchase_dow", "primary_payment_type",
]


def build_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list,
) -> Report:
    """
    Build an Evidently drift report comparing reference vs current.
    Reference = training set sample
    Current   = test set (simulates incoming production data)
    """
    num_cols = [f for f in NUMERICAL_FEATURES if f in features]
    cat_cols = [f for f in CATEGORICAL_FEATURES if f in features]
    use_cols = num_cols + cat_cols

    ref = reference_df[use_cols].copy()
    cur = current_df[use_cols].copy()

    # Cap sample sizes for performance
    ref = ref.sample(min(5000, len(ref)), random_state=42)
    cur = cur.sample(min(5000, len(cur)), random_state=42)

    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftPreset(),
    ])
    report.run(reference_data=ref, current_data=cur)
    return report


def get_drift_summary(report: Report) -> dict:
    """Extract key drift stats from Evidently report."""
    result = report.as_dict()
    metrics = result.get("metrics", [])

    summary = {
        "dataset_drift_detected": False,
        "share_drifted_features": 0.0,
        "n_drifted_features": 0,
        "n_features_checked": 0,
        "drifted_features": [],
    }

    for metric in metrics:
        if metric.get("metric") == "DatasetDriftMetric":
            r = metric.get("result", {})
            summary["dataset_drift_detected"] = r.get("dataset_drift", False)
            summary["share_drifted_features"] = r.get(
                "share_of_drifted_columns", 0.0
            )
            summary["n_drifted_features"] = r.get("number_of_drifted_columns", 0)
            summary["n_features_checked"] = r.get("number_of_columns", 0)

        if metric.get("metric") == "DataDriftTable":
            r = metric.get("result", {})
            drift_by_col = r.get("drift_by_columns", {})
            for col, stats in drift_by_col.items():
                if stats.get("drift_detected", False):
                    summary["drifted_features"].append({
                        "feature"   : col,
                        "p_value"   : round(stats.get("p_value", 0), 4),
                        "stattest"  : stats.get("stattest_name", ""),
                        "drift_score": round(stats.get("drift_score", 0), 4),
                    })

    return summary