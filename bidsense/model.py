"""
Component 2: how many bidders will this tender get?

Loads the small scikit-learn models that scripts/train.py saved in models/ and predicts
from the fields the agent extracted. No training and no network here.
"""

import json
from functools import lru_cache

import joblib
import pandas as pd

import config
from bidsense import features


@lru_cache(maxsize=1)
def _models() -> dict:
    return {name: joblib.load(config.MODELS_DIR / f"bidders_{name}.joblib") for name in ("mean", "q10", "median", "q90")}


@lru_cache(maxsize=1)
def _baseline() -> dict:
    return json.loads((config.MODELS_DIR / "baseline.json").read_text(encoding="utf-8"))


def baseline_bidders(cpv_code) -> float:
    """The baseline to beat: the median number of offers in the tender's CPV division."""
    table = _baseline()
    return float(table["by_division"].get(features.cpv_division(cpv_code) or "", table["overall"]))


def predict(fields: dict) -> dict:
    """Expected bidders with an 80% range, from the agent's fields ({name: value})."""
    x = features.matrix(pd.DataFrame([fields]))
    models = _models()
    point = max(1.0, float(models[config.POINT_MODEL].predict(x)[0]))
    low = max(1.0, min(point, float(models["q10"].predict(x)[0])))
    high = max(point, float(models["q90"].predict(x)[0]))
    return {
        "expected_bidders": round(point, 2),
        "range_low": round(low, 2),
        "range_high": round(high, 2),
        "poisson_mean": round(max(1.0, float(models["mean"].predict(x)[0])), 2),
        "baseline_bidders": baseline_bidders(fields.get("cpv_code")),
        "features_missing": [name for name, value in x.iloc[0].items() if pd.isna(value)],
        "model_version": config.MODEL_VERSION,
    }
