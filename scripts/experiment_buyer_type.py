"""
Experiment: does the buyer's type help the bidders model? `uv run python scripts/experiment_buyer_type.py`

buyer_type is city_council / regional_government / ministry / other (scripts/ted_data.py
derives it from the buyer's name and TED's CAE_TYPE; the agent would read it from the text).

Same protocol for both variants, on the median model (the one the app uses):
settings chosen by training on 2018-2021 and validating on 2022; the DECISION is taken on
the validation year (keep the field only if the validation MAE is lower, with a bootstrap
interval that excludes zero); the final fit on 2018-2022 and the 2023 test are reported
for both, but not used to decide. Writes models/experiment_buyer_type.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import ted_data  # noqa: E402
import train as training  # noqa: E402
from bidsense import features  # noqa: E402

BASE = [name for name in features.FEATURES if name != "buyer_type"]
VARIANTS = {"without_buyer_type": BASE, "with_buyer_type": BASE + ["buyer_type"]}


def main() -> None:
    ted_data.prepare_raw()
    table, _ = ted_data.build_table()
    train = table[table["year"].isin(config.TRAIN_YEARS)]
    valid = table[table["year"] == config.VALID_YEAR]
    final = table[table["year"].isin(config.FINAL_TRAIN_YEARS)]
    test = table[table["year"] == config.TEST_YEAR]

    results, errors = {}, {}
    for name, columns in VARIANTS.items():
        print(f"{name}: choosing settings and fitting...", flush=True)
        model, settings = training.select("quantile", 0.5, columns, features.matrix(train, columns), train["offers"],
                                          features.matrix(valid, columns), valid["offers"])
        errors[name] = np.abs(valid["offers"].to_numpy() - model.predict(features.matrix(valid, columns)))
        n_trees = max(1, round(settings["n_trees"] * len(final) / len(train)))
        final_model = training.new_model("quantile", 0.5, columns, n_trees, settings["params"]).fit(
            features.matrix(final, columns), final["offers"])
        test_errors = np.abs(test["offers"].to_numpy() - final_model.predict(features.matrix(test, columns)))
        results[name] = {"validation_mae": round(float(errors[name].mean()), 4), "test_mae": round(float(test_errors.mean()), 4),
                         "settings": settings}
        errors[name + "_test"] = test_errors

    def gain(key_suffix: str = "") -> dict:
        difference = errors["without_buyer_type" + key_suffix] - errors["with_buyer_type" + key_suffix]
        rng = np.random.default_rng(0)
        boot = [difference[rng.integers(0, len(difference), len(difference))].mean() for _ in range(2000)]
        return {"mae_gain": round(float(difference.mean()), 4), "ci95": [round(float(np.quantile(boot, q)), 4) for q in (0.025, 0.975)]}

    results["validation_gain_with_buyer_type"] = gain()
    results["test_gain_with_buyer_type"] = gain("_test")
    low = results["validation_gain_with_buyer_type"]["ci95"][0]
    results["keep_buyer_type"] = bool(low > 0)
    results["rule"] = "keep only if the validation MAE gain is positive with a 95% bootstrap interval above zero"
    results["buyer_type_share"] = {k: round(float(v), 3) for k, v in table["buyer_type"].value_counts(normalize=True).items()}
    (config.MODELS_DIR / "experiment_buyer_type.json").write_text(json.dumps(results, indent=1))

    for name in VARIANTS:
        print(f"  {name:<20} validation MAE {results[name]['validation_mae']:.4f}   test MAE {results[name]['test_mae']:.4f}")
    for split in ("validation", "test"):
        g = results[f"{split}_gain_with_buyer_type"]
        print(f"  {split} MAE gain from buyer_type: {g['mae_gain']:+.4f} (95% interval {g['ci95'][0]:+.4f} to {g['ci95'][1]:+.4f})")
    print(f"  keep buyer_type: {results['keep_buyer_type']}  ->  models/experiment_buyer_type.json")


if __name__ == "__main__":
    main()
