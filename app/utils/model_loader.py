import os
import json
import pickle
import pandas as pd
import numpy as np


def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__), "../../models/champion_model.pkl"
    )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_metadata():
    meta_path = os.path.join(
        os.path.dirname(__file__),
        "../../data/processed/featured/feature_metadata.json"
    )
    with open(meta_path) as f:
        return json.load(f)


def load_validation_results():
    val_path = os.path.join(
        os.path.dirname(__file__),
        "../../models/validation_results.json"
    )
    with open(val_path) as f:
        return json.load(f)


def load_shap_importance():
    shap_path = os.path.join(
        os.path.dirname(__file__),
        "../../models/shap_importance.json"
    )
    with open(shap_path) as f:
        return json.load(f)


def load_tuning_metadata():
    path = os.path.join(
        os.path.dirname(__file__),
        "../../models/tuning_metadata.json"
    )
    with open(path) as f:
        return json.load(f)


def load_train_data():
    path = os.path.join(
        os.path.dirname(__file__),
        "../../data/processed/featured/train_featured.parquet"
    )
    return pd.read_parquet(path, engine="fastparquet")


def load_test_data():
    path = os.path.join(
        os.path.dirname(__file__),
        "../../data/processed/featured/test_featured.parquet"
    )
    return pd.read_parquet(path, engine="fastparquet")