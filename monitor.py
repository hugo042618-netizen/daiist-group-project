"""
Monitoring: KPIs, quality signals over time, drift against the reference set, alerts.

Drift check: the last WINDOW requests are compared with monitoring/reference_features.csv
(the features of eval/reference.csv) on four features: is_spanish, cpv_division,
log_value and text_length, with the population stability index (PSI). Two alert rules:

1. a feature's PSI is above PSI_ALERT *and* above NOISE_MULTIPLE times the PSI that
   chance alone gives with these sample sizes (with 45 reference inputs and 20 live
   ones, chance alone is already about 0.3, so a bare 0.2 would be a coin toss);
2. more than NON_SPANISH_ALERT of the texts are not in Spanish: the model was trained
   on Spanish tenders only.

Every threshold comes from config.py and is marked TEAM TO SET there.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config

ROOT = Path(__file__).resolve().parent
REFERENCE_FILE = ROOT / "monitoring" / "reference_features.csv"

PSI_BINS = 5                                  # quantile bins of the reference; few, because samples are small
PSI_ALERT = config.PSI_ALERT                  # alert when a feature's PSI exceeds this...
NOISE_MULTIPLE = config.PSI_NOISE_MULTIPLE    # ...and this many times the PSI expected by chance (psi_noise)
NON_SPANISH_ALERT = config.NON_SPANISH_ALERT  # alert when more than this share of texts is not Spanish
WINDOW = config.DRIFT_WINDOW                  # the drift check looks at the last WINDOW requests...
MIN_WINDOW = config.DRIFT_MIN_WINDOW          # ...and stays silent with fewer than this
THUMBS_DOWN_ALERT = config.THUMBS_DOWN_ALERT  # alert when this share of rated requests is 👎
DRIFT_COLUMNS = ["feature", "metric", "value", "threshold", "alert"]


def _bin(values: np.ndarray, reference: np.ndarray, bins: int) -> np.ndarray:
    """Bin index of each value, with bins defined by the reference.

    A feature with few distinct values in the reference (is_spanish) gets one bin per
    value plus one for anything else; a continuous one gets quantile bins, so each
    holds about the same share of the reference.
    """
    distinct = np.unique(reference)
    if len(distinct) <= bins:
        index = np.searchsorted(distinct, values)
        found = (index < len(distinct)) & (distinct[np.minimum(index, len(distinct) - 1)] == values)
        return np.where(found, index, len(distinct))
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(edges, values, side="right")


def psi(reference, current, bins: int = PSI_BINS) -> float:
    """Population stability index of `current` against `reference`.

    0 means the same distribution; the usual rule of thumb is < 0.1 stable, 0.1-0.25
    moderate, > 0.25 a large shift. With small samples PSI is large by chance: see `psi_noise`.
    """
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    reference, current = reference[~np.isnan(reference)], current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    ref_bins, cur_bins = _bin(reference, reference, bins), _bin(current, reference, bins)
    n_bins = max(ref_bins.max(), cur_bins.max()) + 1
    # Half a count is added to every bin: with 10-20 live inputs an empty bin is common by
    # chance, and without smoothing a single empty bin alone would push PSI above 1.
    expected = (np.bincount(ref_bins, minlength=n_bins) + 0.5) / (len(reference) + 0.5 * n_bins)
    actual = (np.bincount(cur_bins, minlength=n_bins) + 0.5) / (len(current) + 0.5 * n_bins)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_noise(n_reference: int, n_current: int, bins: int = PSI_BINS) -> float:
    """PSI you would expect by chance, with no drift at all: about (bins - 1) * (1/n_reference + 1/n_current)."""
    return (bins - 1) * (1 / n_reference + 1 / n_current)


def load_reference() -> pd.DataFrame | None:
    """Features of the reference inputs (`main.py reference` writes them)."""
    return pd.read_csv(REFERENCE_FILE) if REFERENCE_FILE.is_file() else None


def drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """One row per drift check, and whether it alerts. Both frames have one column per feature (no `f_` prefix)."""
    rows = []
    for feature in config.DRIFT_FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            continue
        n_reference, n_current = int(reference[feature].notna().sum()), int(current[feature].notna().sum())
        if n_current == 0:
            rows.append({"feature": feature, "metric": "PSI", "value": np.nan, "threshold": PSI_ALERT, "alert": False})
            continue
        value = psi(reference[feature], current[feature])
        bins = min(PSI_BINS, reference[feature].nunique() + 1)
        threshold = max(PSI_ALERT, NOISE_MULTIPLE * psi_noise(n_reference, n_current, bins))
        rows.append({"feature": feature, "metric": "PSI", "value": round(value, 3), "threshold": round(threshold, 3),
                     "alert": bool(value > threshold)})
    if "is_spanish" in current.columns and current["is_spanish"].notna().any():
        share = float(1 - current["is_spanish"].mean())
        rows.append({"feature": "texts not in Spanish", "metric": "share", "value": round(share, 3),
                     "threshold": NON_SPANISH_ALERT, "alert": bool(share > NON_SPANISH_ALERT)})
    return pd.DataFrame(rows, columns=DRIFT_COLUMNS)


def live_features(logs: pd.DataFrame) -> pd.DataFrame:
    """The features of the last WINDOW logged requests, without the `f_` prefix."""
    columns = [c for c in logs.columns if c.startswith("f_")]
    return logs[columns].tail(WINDOW).rename(columns=lambda c: c[2:])


def quality(logs: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Quality signals over time, in long format (time, signal, value) for a line plot.

    One point per request: the average over the last `window` requests (for thumbs
    down, the last `window` rated requests). With little traffic, hourly averages would
    be a single dot; a rolling average shows the trend from the first requests on.
    """
    if logs.empty:
        return pd.DataFrame(columns=["time", "signal", "value"])
    df = logs.sort_values("timestamp")
    signals = pd.DataFrame({"time": df["timestamp"]})
    signals["share sent to review"] = df["review"].astype(float).rolling(window, min_periods=1).mean()
    if "f_confidence" in df:
        signals["mean confidence"] = df["f_confidence"].astype(float).rolling(window, min_periods=1).mean()
    rated = df["feedback"].astype(float)
    signals["share 👎 (of rated)"] = (rated < 0).where(rated.notna()).astype(float).dropna().rolling(window, min_periods=1).mean()
    long = signals.melt("time", var_name="signal", value_name="value")
    return long.dropna().reset_index(drop=True)


def kpis(logs: pd.DataFrame) -> dict:
    """The numbers of the KPI tiles."""
    if logs.empty:
        return {"requests": 0, "thumbs_down_share": None, "rated": 0, "review_share": None, "mean_cost_eur": None,
                "mean_latency_s": None, "total_cost_eur": 0.0}
    rated = logs["feedback"].dropna().astype(float)
    return {
        "requests": int(len(logs)),
        "rated": int(len(rated)),
        "thumbs_down_share": float((rated < 0).mean()) if len(rated) else None,
        "review_share": float(logs["review"].mean()),
        "mean_cost_eur": float(logs["cost_eur"].mean()),
        "mean_latency_s": float(logs["latency_s"].mean()),
        "total_cost_eur": float(logs["cost_eur"].sum()),
    }


def summary(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame({"metric": ["requests"], "value": [0]})
    k = kpis(logs)
    rows = {
        "requests": k["requests"],
        "sent to review": f"{k['review_share']:.0%}",
        "rated 👍/👎": k["rated"],
        "share 👎 (of rated)": f"{k['thumbs_down_share']:.0%}" if k["rated"] else "-",
        "mean latency (s)": round(k["mean_latency_s"], 2),
        "mean cost per request (€)": round(k["mean_cost_eur"], 5),
        "total cost (€)": round(k["total_cost_eur"], 4),
    }
    return pd.DataFrame({"metric": list(rows), "value": [str(v) for v in rows.values()]})


def recent(logs: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """The latest requests as summaries only: never the tender text."""
    columns = ["time (UTC)", "input summary", "decision", "expected bidders", "confidence", "latency (s)", "cost (€)", "feedback"]
    if logs.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for record in logs.tail(n).iloc[::-1].to_dict("records"):
        outputs = record.get("outputs") if isinstance(record.get("outputs"), dict) else {}
        prediction = outputs.get("model") or {}
        feedback = record.get("feedback")
        rows.append({
            "time (UTC)": record["timestamp"].strftime("%d %b %H:%M"),
            "input summary": record.get("input_summary", ""),
            "decision": record.get("decision") or (outputs.get("decision") or {}).get("decision", ""),
            "expected bidders": prediction.get("expected_bidders", ""),
            "confidence": record.get("f_confidence", ""),
            "latency (s)": record.get("latency_s", ""),
            "cost (€)": round(float(record.get("cost_eur", 0)), 5),
            "feedback": "" if pd.isna(feedback) else ("👍" if float(feedback) > 0 else "👎"),
        })
    return pd.DataFrame(rows, columns=columns)


def alerts(logs: pd.DataFrame, reference: pd.DataFrame | None) -> list[str]:
    """The alert rules. Each fired alert is one message."""
    fired = []
    window = logs.tail(WINDOW)
    if reference is not None and len(window) >= MIN_WINDOW:
        report = drift(reference, live_features(window))
        for row in report[report["alert"]].itertuples():
            fired.append(f"Drift in `{row.feature}`: {row.metric} {row.value} > {row.threshold} over the last {len(window)} requests.")
    rated = window["feedback"].dropna().astype(float) if not window.empty else pd.Series(dtype=float)
    if len(rated) >= MIN_WINDOW and (rated < 0).mean() > THUMBS_DOWN_ALERT:
        fired.append(f"{(rated < 0).mean():.0%} of the last {len(rated)} rated requests are 👎 (> {THUMBS_DOWN_ALERT:.0%}).")
    return fired
