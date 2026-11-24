"""
Rebuild the bidders model from scratch. `uv run python scripts/train.py`

1. downloads the TED CSV files and the CPV list if they are missing (data/raw/, git-ignored);
2. builds the Spanish training table (scripts/ted_data.py);
3. trains, on the older years: a Poisson model for the expected number of offers and
   quantile models for the median and the 80% range; picks settings on the validation year;
4. reports the test year against the baseline (median offers per CPV division);
5. writes models/*.joblib, models/metrics.json and data/market_stats.json.

The first run downloads about 1.1 GB and takes a few minutes. The app never runs
this: it only loads the files in models/ and data/.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_poisson_deviance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import ted_data  # noqa: E402
from bidsense import features  # noqa: E402

SEED = 0
CATEGORICAL_MASK = [name in features.CATEGORICAL for name in features.FEATURES]
GRID = [
    {"max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 1.0},
    {"max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0},
    {"max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 5.0},
    {"max_leaf_nodes": 63, "min_samples_leaf": 100, "l2_regularization": 5.0},
]
MAX_ITER, LEARNING_RATE = 600, 0.05


def bucket(offers) -> np.ndarray:
    """1-2, 3-5 or 6+ bidders (config.BIDDER_BUCKETS)."""
    offers = np.rint(np.asarray(offers, float))
    labels = np.empty(len(offers), dtype=object)
    for low, high, name in config.BIDDER_BUCKETS:
        labels[(offers >= low) & (offers <= (high if high is not None else np.inf))] = name
    labels[offers < config.BIDDER_BUCKETS[0][0]] = config.BIDDER_BUCKETS[0][2]
    return labels


def fit(loss: str, quantile, x_train, y_train, x_valid, y_valid, score) -> tuple[HistGradientBoostingRegressor, dict]:
    """Try each setting in GRID, choose the number of trees and the setting on the validation year."""
    best = None
    for params in GRID:
        model = HistGradientBoostingRegressor(
            loss=loss, quantile=quantile, learning_rate=LEARNING_RATE, max_iter=MAX_ITER,
            categorical_features=CATEGORICAL_MASK, early_stopping=False, random_state=SEED, **params,
        ).fit(x_train, y_train)
        scores = [score(y_valid, prediction) for prediction in model.staged_predict(x_valid)]
        n_trees = int(np.argmin(scores)) + 1
        if best is None or scores[n_trees - 1] < best[0]:
            best = (scores[n_trees - 1], params, n_trees)
    _, params, n_trees = best
    model = HistGradientBoostingRegressor(
        loss=loss, quantile=quantile, learning_rate=LEARNING_RATE, max_iter=n_trees,
        categorical_features=CATEGORICAL_MASK, early_stopping=False, random_state=SEED, **params,
    ).fit(x_train, y_train)
    return model, params | {"n_trees": n_trees, "validation_score": round(float(best[0]), 4)}


def evaluate(y, predictions: dict[str, np.ndarray], low, high) -> dict:
    y = np.asarray(y, float)
    report = {"n": int(len(y))}
    for name, prediction in predictions.items():
        report[name] = {
            "mae": round(float(mean_absolute_error(y, prediction)), 4),
            "rmse": round(float(np.sqrt(np.mean((y - prediction) ** 2))), 4),
            "poisson_deviance": round(float(mean_poisson_deviance(y, np.clip(prediction, 0.1, None))), 4),
            "bucket_accuracy": round(float((bucket(y) == bucket(prediction)).mean()), 4),
        }
    report["range_80"] = {
        "coverage": round(float(((y >= low) & (y <= high)).mean()), 4),
        "share_below_low": round(float((y < low).mean()), 4),
        "share_above_high": round(float((y > high).mean()), 4),
        "mean_width": round(float((high - low).mean()), 2),
    }
    return report


def market_stats(train: pd.DataFrame) -> dict:
    """Quartiles of offers per CPV division and contract type, from the training years.

    This is what the agent's `market_stats` tool returns. Keys: "45|works", and "45" for
    the whole division.
    """
    table = train.assign(division=train["cpv_code"].str[:2])
    stats = {}
    for keys, group in [(("division",), None), (("division", "contract_type"), None)]:
        for key, offers in table.groupby(list(keys))["offers"]:
            key = key if isinstance(key, tuple) else (key,)
            if len(offers) >= 20:
                q25, q50, q75 = np.quantile(offers, [0.25, 0.5, 0.75])
                stats["|".join(key)] = {"n": int(len(offers)), "q25": float(q25), "median": float(q50), "q75": float(q75)}
    q25, q50, q75 = np.quantile(table["offers"], [0.25, 0.5, 0.75])
    stats["all"] = {"n": int(len(table)), "q25": float(q25), "median": float(q50), "q75": float(q75)}
    return stats


def main() -> None:
    ted_data.prepare_raw()
    table, counts = ted_data.build_table()
    print(f"\n{counts['can_notices']:,} Spanish award notices; {counts['usable_notices']:,} usable "
          f"({counts['share_notices_valid_offers']:.1%} of those open to competition have a valid number of offers).")

    train = table[table["year"].isin(config.TRAIN_YEARS)]
    valid = table[table["year"] == config.VALID_YEAR]
    test = table[table["year"] == config.TEST_YEAR]
    x_train, x_valid, x_test = (features.matrix(part) for part in (train, valid, test))
    y_train, y_valid, y_test = train["offers"], valid["offers"], test["offers"]
    print(f"train {min(config.TRAIN_YEARS)}-{max(config.TRAIN_YEARS)}: {len(train):,}   "
          f"validation {config.VALID_YEAR}: {len(valid):,}   test {config.TEST_YEAR}: {len(test):,}")

    # Baseline: the median number of offers in the tender's CPV division, from the training years.
    overall = float(y_train.median())
    by_division = y_train.groupby(train["cpv_code"].str[:2]).median().to_dict()
    baseline = lambda part: part["cpv_code"].str[:2].map(by_division).fillna(overall).to_numpy()  # noqa: E731

    print("Training (4 settings x 4 models)...", flush=True)
    models, settings = {}, {}
    models["mean"], settings["mean"] = fit("poisson", None, x_train, y_train, x_valid, y_valid, mean_poisson_deviance)
    for name, q in (("q10", config.QUANTILE_LOW), ("median", 0.5), ("q90", config.QUANTILE_HIGH)):
        models[name], settings[name] = fit(
            "quantile", q, x_train, y_train, x_valid, y_valid, lambda y, p, q=q: mean_pinball_loss(y, p, alpha=q))

    def report(x, y, part) -> dict:
        p = {name: model.predict(x) for name, model in models.items()}
        return evaluate(y, {"baseline_median_per_cpv_division": baseline(part), "model_median": p["median"],
                            "model_poisson_mean": p["mean"]}, p["q10"], p["q90"])

    metrics = {
        "model_version": config.MODEL_VERSION,
        "sklearn_version": sklearn.__version__,
        "features": features.FEATURES,
        "split": {"train_years": list(config.TRAIN_YEARS), "validation_year": config.VALID_YEAR, "test_year": config.TEST_YEAR},
        "data": counts,
        "settings": settings,
        "validation": report(x_valid, y_valid, valid),
        "test": report(x_test, y_test, test),
    }

    config.MODELS_DIR.mkdir(exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, config.MODELS_DIR / f"bidders_{name}.joblib", compress=3)
    (config.MODELS_DIR / "baseline.json").write_text(json.dumps({"overall": overall, "by_division": by_division}, indent=1))
    (config.MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (config.DATA_DIR / "market_stats.json").write_text(json.dumps(market_stats(train), indent=1))

    print(f"\nTest year {config.TEST_YEAR} ({metrics['test']['n']:,} notices):")
    for name in ("baseline_median_per_cpv_division", "model_median", "model_poisson_mean"):
        row = metrics["test"][name]
        print(f"  {name:<34} MAE {row['mae']:.3f}   RMSE {row['rmse']:.3f}   Poisson deviance {row['poisson_deviance']:.3f}   "
              f"bucket accuracy {row['bucket_accuracy']:.1%}")
    spread = metrics["test"]["range_80"]
    print(f"  80% range: covers {spread['coverage']:.1%} of cases ({spread['share_below_low']:.1%} below, "
          f"{spread['share_above_high']:.1%} above), {spread['mean_width']} bidders wide on average")
    sizes = sum(p.stat().st_size for p in config.MODELS_DIR.glob("*.joblib")) / 1e6
    print(f"\nWrote models/ ({sizes:.1f} MB of models), models/metrics.json and data/market_stats.json.")


if __name__ == "__main__":
    main()
