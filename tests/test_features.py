import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.tabs.user_tab import build_feature_vector

MOCK_META = {
    "final_features": [
        "purchase_dow", "purchase_month", "purchase_quarter",
        "purchase_hour", "purchase_is_weekend", "purchase_year",
        "purchase_near_holiday", "purchase_black_friday",
        "estimated_delivery_days", "item_count", "total_price",
        "total_freight", "avg_price", "unique_sellers", "unique_products",
        "product_weight_g", "product_length_cm", "product_height_cm",
        "product_width_cm", "product_volume_cm3", "volumetric_weight",
        "product_name_lenght", "product_description_lenght",
        "product_photos_qty", "freight_per_kg", "freight_price_ratio",
        "price_per_item", "is_heavy_product", "is_bulky_product",
        "total_payment_value", "payment_installments",
        "payment_methods_used", "same_state", "same_region",
        "cross_region", "customer_state_avg_variance",
        "seller_state_avg_variance", "seller_avg_variance",
        "seller_std_variance", "seller_order_count", "seller_late_rate",
        "customer_order_count", "is_repeat_customer", "category_avg_variance",
    ],
    "target"              : "lead_time_variance",
    "global_seller_avg"   : -11.24,
    "global_seller_std"   : 9.66,
    "global_seller_late"  : 0.08,
    "global_cat_avg"      : -11.24,
}

MOCK_INPUTS = {
    "customer_state"         : "SP",
    "seller_state"           : "SP",
    "estimated_delivery_days": 23,
    "price"                  : 150.0,
    "freight"                : 20.0,
    "weight_g"               : 500,
    "item_count"             : 1,
    "length_cm"              : 25.0,
    "height_cm"              : 13.0,
    "width_cm"               : 20.0,
    "photos_qty"             : 2,
    "product_name_length"    : 50,
    "product_desc_length"    : 600,
    "purchase_month"         : 6,
    "purchase_dow"           : 2,
    "purchase_hour"          : 14,
    "purchase_year"          : 2018,
    "purchase_near_holiday"  : 0,
    "purchase_black_friday"  : 0,
    "payment_type"           : "credit_card",
    "payment_installments"   : 1,
}


def test_feature_vector_shape():
    fv = build_feature_vector(MOCK_INPUTS, MOCK_META)
    assert fv.shape == (1, 44), f"Expected (1, 44), got {fv.shape}"


def test_feature_vector_no_nulls():
    fv = build_feature_vector(MOCK_INPUTS, MOCK_META)
    assert fv.isnull().sum().sum() == 0, "Feature vector contains nulls"


def test_same_state_flag():
    fv = build_feature_vector(MOCK_INPUTS, MOCK_META)
    assert fv["same_state"].iloc[0] == 1, "SP→SP should be same_state=1"


def test_cross_state_flag():
    inputs = MOCK_INPUTS.copy()
    inputs["customer_state"] = "RJ"
    inputs["seller_state"]   = "SP"
    fv = build_feature_vector(inputs, MOCK_META)
    assert fv["same_state"].iloc[0] == 0, "RJ→SP should be same_state=0"


def test_is_weekend_flag():
    inputs = MOCK_INPUTS.copy()
    inputs["purchase_dow"] = 5  # Saturday
    fv = build_feature_vector(inputs, MOCK_META)
    assert fv["purchase_is_weekend"].iloc[0] == 1


def test_heavy_product_flag():
    inputs = MOCK_INPUTS.copy()
    inputs["weight_g"] = 6000
    fv = build_feature_vector(inputs, MOCK_META)
    assert fv["is_heavy_product"].iloc[0] == 1


def test_feature_columns_match_meta():
    fv = build_feature_vector(MOCK_INPUTS, MOCK_META)
    assert list(fv.columns) == MOCK_META["final_features"]