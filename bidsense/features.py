"""
The features of the bidders model, built from the fields the agent extracts.

Training (scripts/train.py) and the app (pipeline.py) both call `matrix()`, so a
tender is encoded in exactly the same way in both places. The model only sees
what the agent can read in a tender text: nothing that is known only after the
award.
"""

import math

import numpy as np
import pandas as pd

# The 45 divisions of CPV 2008 (first two digits of a code).
CPV_DIVISIONS = [
    "03", "09", "14", "15", "16", "18", "19", "22", "24", "30", "31", "32", "33", "34", "35",
    "37", "38", "39", "41", "42", "43", "44", "45", "48", "50", "51", "55", "60", "63", "64",
    "65", "66", "70", "71", "72", "73", "75", "76", "77", "79", "80", "85", "90", "92", "98",
]

CONTRACT_TYPES = ["works", "supplies", "services"]
PROCEDURES = ["open", "restricted", "negotiated", "other"]

# Spanish regions as NUTS-2 codes (comunidades and ciudades autónomas).
REGIONS = {
    "ES11": "Galicia", "ES12": "Asturias", "ES13": "Cantabria", "ES21": "País Vasco",
    "ES22": "Navarra", "ES23": "La Rioja", "ES24": "Aragón", "ES30": "Madrid",
    "ES41": "Castilla y León", "ES42": "Castilla-La Mancha", "ES43": "Extremadura",
    "ES51": "Cataluña", "ES52": "Comunitat Valenciana", "ES53": "Illes Balears",
    "ES61": "Andalucía", "ES62": "Murcia", "ES63": "Ceuta", "ES64": "Melilla", "ES70": "Canarias",
}

BUYER_TYPES = ["city_council", "regional_government", "ministry", "other"]

# buyer_type was added after an experiment (scripts/experiment_buyer_type.py, models/experiment_buyer_type.json).
CATEGORICAL = ["cpv_division", "contract_type", "procedure", "region", "buyer_type"]
ALL_CATEGORICAL = CATEGORICAL
NUMERIC = ["cpv_group", "log_value", "duration_months", "lots_number", "price_weight_pct", "is_framework_agreement"]
FEATURES = CATEGORICAL + NUMERIC

# Values below this are placeholders in practice ("1 euro"), not real estimates.
MIN_VALUE_EUR = 1_000


def cpv_division(cpv_code) -> str | None:
    """First two digits of a CPV code, if it is one of the 45 real divisions."""
    digits = "".join(c for c in str(cpv_code or "") if c.isdigit())
    return digits[:2] if len(digits) >= 2 and digits[:2] in CPV_DIVISIONS else None


def _number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _code(values: pd.Series, vocabulary: list[str]) -> pd.Series:
    return values.map({name: float(i) for i, name in enumerate(vocabulary)}).astype(float)


def matrix(fields: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """One row of model features per row of agent fields. Missing or invalid values become NaN.

    `columns` defaults to FEATURES, the model's features; experiments pass their own list.

    Expected columns (all optional): cpv_code, contract_type, procedure, region,
    estimated_value_eur, duration_months, lots_number, price_weight_pct,
    is_framework_agreement, buyer_type.
    """
    get = lambda name: fields[name] if name in fields else pd.Series([None] * len(fields), index=fields.index)  # noqa: E731
    out = pd.DataFrame(index=fields.index)

    digits = get("cpv_code").map(lambda c: "".join(ch for ch in str(c or "") if ch.isdigit()))
    division = digits.map(cpv_division)
    out["cpv_division"] = pd.to_numeric(division, errors="coerce")           # 3..98, treated as a category
    out["contract_type"] = _code(get("contract_type"), CONTRACT_TYPES)
    out["procedure"] = _code(get("procedure"), PROCEDURES)
    out["region"] = _code(get("region"), list(REGIONS))

    out["cpv_group"] = pd.to_numeric(digits.str[:3].where(division.notna()), errors="coerce")
    value = get("estimated_value_eur").map(_number)
    out["log_value"] = np.log10(value.where(value >= MIN_VALUE_EUR))
    duration = get("duration_months").map(_number)
    out["duration_months"] = duration.where(duration > 0).clip(upper=240)
    lots = get("lots_number").map(_number)
    out["lots_number"] = lots.where(lots >= 1).clip(upper=200)
    weight = get("price_weight_pct").map(_number)
    out["price_weight_pct"] = weight.where((weight >= 0) & (weight <= 100))
    out["is_framework_agreement"] = get("is_framework_agreement").map(
        lambda v: math.nan if v is None or (isinstance(v, float) and math.isnan(v)) else float(bool(v))
    )
    out["buyer_type"] = _code(get("buyer_type"), BUYER_TYPES)
    return out[columns or FEATURES].astype(float)


def encode(fields: dict) -> dict[str, float]:
    """The features of one tender, as {name: number}."""
    return matrix(pd.DataFrame([fields])).iloc[0].to_dict()
