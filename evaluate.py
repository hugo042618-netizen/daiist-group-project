"""
Evaluation. `uv run python main.py evaluate`

1. Component 2 (bidders model): MAE and bucket accuracy (1-2, 3-5, 6+ bidders) on the test
   year against the baseline, read from models/metrics.json (scripts/train.py wrote it).
2. Component 1 (agent): field accuracy against TED's official fields for the same notices
   (eval/extraction_truth.csv), and how much the hand-off costs: the model's error with the
   agent's fields versus with the official fields.
3. End to end: every row of eval/cases.csv, compared with `expected` and with the baseline.
4. LLM judge: Gemini judges whether the reasons match the tender; agreement with the team's
   labels in eval/human_labels.csv (Cohen's kappa).
5. Cost per request: tokens, euros, seconds.

The agent and the judge are measured on the app's default model only (config.GEMINI_MODEL):
fallback models are not used here. Every Gemini call is cached on disk (cache/), so a second
run costs nothing. Results go to
eval/results.csv, eval/extraction_results.csv and eval/metrics.json: commit them, so the
numbers in REPORT.md can be traced back.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
import env
import pipeline
from bidsense import agent, decision, judge, model
from bidsense.redact import redact

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "eval" / "cases.csv"
HUMAN_LABELS = ROOT / "eval" / "human_labels.csv"
TRUTH = ROOT / "eval" / "extraction_truth.csv"
RESULTS = ROOT / "eval" / "results.csv"
EXTRACTION_RESULTS = ROOT / "eval" / "extraction_results.csv"
METRICS = ROOT / "eval" / "metrics.json"

EXTRACTION_SAMPLE = 40   # notices of extraction_truth.csv to run (each is one agent run on the free tier)
DECISIONS = {
    "bid": decision.BID, "dontbid": decision.DONT_BID, "nobid": decision.DONT_BID, "donotbid": decision.DONT_BID,
    "review": decision.REVIEW, "sendtoreview": decision.REVIEW, "outofscope": decision.OUT_OF_SCOPE,
}


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]


def as_decision(text: str) -> str | None:
    """'DON'T BID', 'dont_bid', 'Send to review'... -> the product's decision label."""
    return DECISIONS.get("".join(c for c in str(text).lower() if c.isalpha()))


def is_correct(case: dict, result: dict) -> bool | None:
    """Is the end-to-end decision the one the team expected? None if `expected` is not a decision label."""
    expected = as_decision(case["expected"])
    return None if expected is None else result["decision"] == expected


def bucket(offers: float) -> str:
    offers = round(float(offers))
    return next(name for low, high, name in config.BIDDER_BUCKETS if offers <= (high if high is not None else np.inf))


# --- 1. Component 2: the bidders model ------------------------------------------

def evaluate_model() -> dict:
    metrics = json.loads((config.MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
    test = metrics["test"]
    print(f"\n=== Component 2: bidders model, test year {metrics['split']['test_year']} ({test['n']:,} notices) ===")
    for name, label in (("baseline_median_per_cpv_division", "baseline: median per CPV division"),
                        ("model_median", "model (median, used by the app)"), ("model_poisson_mean", "model (Poisson mean)")):
        print(f"  {label:<36} MAE {test[name]['mae']:.3f}   bucket accuracy {test[name]['bucket_accuracy']:.1%}")
    return {"test_year": metrics["split"]["test_year"], "n": test["n"], **{k: test[k] for k in test if k != "n"}}


# --- 2. Component 1: the agent ---------------------------------------------------

def _close(a, b, tolerance: float, relative: bool = False) -> bool:
    return abs(float(a) - float(b)) <= (tolerance * abs(float(b)) if relative else tolerance)


FIELD_CHECKS = {
    "cpv_division": lambda got, truth: str(got)[:2] == str(truth["cpv_code"]).zfill(8)[:2],
    "cpv_group (3 digits)": lambda got, truth: str(got)[:3] == str(truth["cpv_code"]).zfill(8)[:3],
    "contract_type": lambda got, truth: got == truth["contract_type"],
    "estimated_value_eur (within 1%)": lambda got, truth: _close(got, truth["estimated_value_eur"], 0.01, relative=True),
    "duration_months (within 1)": lambda got, truth: _close(got, truth["duration_months"], 1.0),
    "procedure": lambda got, truth: got == truth["procedure"],
    "lots_number": lambda got, truth: int(got) == int(float(truth["lots_number"])),
    "price_weight_pct (within 2)": lambda got, truth: _close(got, truth["price_weight_pct"], 2.0),
    "is_framework_agreement": lambda got, truth: bool(got) == (str(truth["is_framework_agreement"]).lower() == "true"),
    "region": lambda got, truth: got == truth["region"],
    "buyer_type": lambda got, truth: got == truth["buyer_type"],
}
TRUTH_COLUMN = {"cpv_division": "cpv_code", "cpv_group (3 digits)": "cpv_code"}
AGENT_FIELD = {"cpv_division": "cpv_code", "cpv_group (3 digits)": "cpv_code"}


def evaluate_agent() -> dict:
    truth_rows = read_rows(TRUTH)[:EXTRACTION_SAMPLE]
    print(f"\n=== Component 1: agent, field accuracy on {len(truth_rows)} notices of eval/extraction_truth.csv ===")
    if not truth_rows:
        print("  eval/extraction_truth.csv is empty: run scripts/fetch_candidates.py.")
        return {}
    rows, errors = [], 0
    for i, truth in enumerate(truth_rows, 1):
        print(f"  {i}/{len(truth_rows)}", end="\r", flush=True)
        try:
            extraction = agent.extract(redact(truth["input"])[0], default_model_only=True)
        except agent.AgentError as error:
            errors += 1
            if errors == 1:
                print(f"  agent error: {error}")
            continue
        values = {name: field["value"] for name, field in extraction["fields"].items()}
        row = {"id": truth["id"], "model": extraction.get("model", ""), "is_tender": extraction["is_tender"],
               "offers": float(truth["offers"])}
        for check, test in FIELD_CHECKS.items():
            name = check.split(" ")[0]
            expected = truth.get(TRUTH_COLUMN.get(check, name), "")
            got = values.get(AGENT_FIELD.get(check, name))
            row[check] = None if expected in ("", None) else (got is not None and bool(test(got, truth)))
        official = {"cpv_code": str(truth["cpv_code"]).zfill(8), "contract_type": truth["contract_type"], "procedure": truth["procedure"],
                    "region": truth["region"], "estimated_value_eur": truth["estimated_value_eur"] or None,
                    "duration_months": truth["duration_months"] or None, "lots_number": truth["lots_number"] or None,
                    "price_weight_pct": truth["price_weight_pct"] or None,
                    "is_framework_agreement": str(truth["is_framework_agreement"]).lower() == "true",
                    "buyer_type": truth.get("buyer_type") or None}
        row["bidders_from_agent_fields"] = model.predict(values)["expected_bidders"]
        row["bidders_from_official_fields"] = model.predict(official)["expected_bidders"]
        row["bidders_baseline"] = model.baseline_bidders(official["cpv_code"])
        rows.append(row | {"tokens_in": extraction["tokens_in"], "tokens_out": extraction["tokens_out"], "agent_latency_s": extraction["latency_s"]})
    if not rows:
        print(f"  no notice could be read ({errors} agent errors). Is GEMINI_API_KEY set?")
        return {"errors": errors}
    table = pd.DataFrame(rows)
    table.to_csv(EXTRACTION_RESULTS, index=False)
    accuracy = {}
    for check in FIELD_CHECKS:
        scored = table[check].dropna()
        if len(scored):
            accuracy[check] = {"accuracy": round(float(scored.astype(float).mean()), 4), "n": int(len(scored))}
            print(f"  {check:<34} {accuracy[check]['accuracy']:.0%}  (n={len(scored)})")
    print(f"  recognised as a tender: {table['is_tender'].mean():.0%}" + (f"   ({errors} agent errors)" if errors else ""))
    handoff = {name: round(float((table[column] - table["offers"]).abs().mean()), 3) for name, column in
               (("mae_with_agent_fields", "bidders_from_agent_fields"), ("mae_with_official_fields", "bidders_from_official_fields"),
                ("mae_baseline", "bidders_baseline"))}
    handoff["bucket_accuracy_with_agent_fields"] = round(float(np.mean(
        [bucket(p) == bucket(y) for p, y in zip(table["bidders_from_agent_fields"], table["offers"])])), 3)
    print(f"  Hand-off on these {len(table)} notices from 2026 (real offers known): MAE of expected bidders "
          f"{handoff['mae_with_agent_fields']} with the agent's fields, {handoff['mae_with_official_fields']} with the official fields, "
          f"{handoff['mae_baseline']} for the baseline.")
    return {"n": int(len(table)), "errors": errors, "is_tender_rate": round(float(table["is_tender"].mean()), 4),
            "field_accuracy": accuracy, "handoff": handoff}


# --- 3-5. End to end, judge, cost ------------------------------------------------

def evaluate_cases() -> dict:
    cases = [c for c in read_rows(CASES) if (c.get("input") or "").strip()]
    print(f"\n=== End to end: {len(cases)} cases of eval/cases.csv ===")
    if not cases:
        print("  eval/cases.csv has no cases yet. The team writes them (see eval/candidates/README.md).")
        return {}
    squash = lambda text: " ".join(str(text).split())  # noqa: E731
    reference = {squash(r.get("input", "")) for r in read_rows(ROOT / "eval" / "reference.csv")}
    if reused := [c["id"] for c in cases if squash(c["input"]) in reference]:
        print(f"  WARNING: {len(reused)} cases are also in eval/reference.csv ({', '.join(reused[:8])}): the reference set must not hold test cases.")
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"  {i}/{len(cases)}", end="\r", flush=True)
        started = time.perf_counter()
        result = pipeline.run(case["input"], default_model_only=True)
        elapsed = time.perf_counter() - started
        outputs = result["outputs"]
        agent_output = outputs["agent"]
        answer = {"recommendation": result["decision"], "reason": outputs["decision"]["reason"],
                  "fields": {n: f["value"] for n, f in (agent_output.get("fields") or {}).items()}}
        verdict = {}
        if not agent_output.get("error"):
            try:
                verdict = judge.judge(redact(pipeline.read_input(case["input"]))[0], answer, default_model_only=True)
            except agent.AgentError as error:
                verdict = {"error": str(error)}
        baseline_decision = (outputs["baseline"] or {}).get("decision") or result["decision"]
        expected = as_decision(case["expected"])
        rows.append({
            "id": case["id"],
            "kind": case["kind"].strip().lower(),
            "expected": expected or case["expected"],
            "decision": result["decision"],
            "review": result["review"],
            "correct": is_correct(case, result),
            "baseline_today": as_decision(case.get("baseline", "")) or case.get("baseline", ""),
            "baseline_today_correct": None if expected is None or as_decision(case.get("baseline", "")) is None
            else as_decision(case["baseline"]) == expected,
            "baseline_median_decision": baseline_decision,
            "baseline_median_correct": None if expected is None else baseline_decision == expected,
            "expected_bidders": (outputs["model"] or {}).get("expected_bidders"),
            "confidence": result["features"]["confidence"],
            "judge_label": verdict.get("label"),
            "judge_explanation": verdict.get("explanation", verdict.get("error", "")),
            "latency_s": round(agent_output.get("latency_s", elapsed) if result["cached"] else elapsed, 3),
            "tokens_in": result["tokens_in"],
            "tokens_out": result["tokens_out"],
            "tokens": result["tokens"],
            "cost_eur": result["cost_eur"],
            "error": agent_output.get("error", ""),
            "agent_model": agent_output.get("model", ""),
            "reason": outputs["decision"]["reason"],
        })
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS, index=False)
    print(f"  Wrote {RESULTS.relative_to(ROOT)} ({len(results)} cases).\n")

    numeric = lambda column: pd.to_numeric(results[column], errors="coerce")  # noqa: E731
    by_kind = results.assign(correct=numeric("correct"), baseline_today=numeric("baseline_today_correct"),
                             baseline_median=numeric("baseline_median_correct")).groupby("kind").agg(
        cases=("id", "count"), sent_to_review=("review", "mean"), product_correct=("correct", "mean"),
        baseline_today_correct=("baseline_today", "mean"), baseline_median_correct=("baseline_median", "mean"),
    )
    print(by_kind.round(3).to_string())
    summary = {"n": int(len(results)), "errors": int((results["error"] != "").sum()),
               "review_rate": round(float(results["review"].mean()), 4)}
    for name, column in (("product_accuracy", "correct"), ("baseline_today_accuracy", "baseline_today_correct"),
                         ("baseline_median_accuracy", "baseline_median_correct")):
        scored = numeric(column).dropna()
        summary[name] = round(float(scored.mean()), 4) if len(scored) else None
    print(f"\n  Product {summary['product_accuracy']}, baseline today (cases.csv) {summary['baseline_today_accuracy']}, "
          f"baseline median-per-CPV-division rule {summary['baseline_median_accuracy']} (share of cases with the expected decision).")
    if summary["errors"]:
        print(f"  WARNING: the agent failed on {summary['errors']} cases (they count as REVIEW). First error: "
              f"{results.loc[results['error'] != '', 'error'].iloc[0][:140]}")

    summary["judge"] = judge_agreement(results)
    summary["cost_per_request"] = {
        "tokens_in": round(float(results["tokens_in"].mean()), 1), "tokens_out": round(float(results["tokens_out"].mean()), 1),
        "cost_eur": round(float(results["cost_eur"].mean()), 6), "latency_s_mean": round(float(results["latency_s"].mean()), 2),
        "latency_s_median": round(float(results["latency_s"].median()), 2), "total_cost_eur": round(float(results["cost_eur"].sum()), 4),
    }
    cost = summary["cost_per_request"]
    print(f"\n=== Cost per request ===\n  {cost['tokens_in']:.0f} tokens in, {cost['tokens_out']:.0f} out, €{cost['cost_eur']:.5f} "
          f"(paid-tier prices; free on the free tier), {cost['latency_s_mean']} s on average (median {cost['latency_s_median']} s).")
    return summary


def judge_agreement(results: pd.DataFrame) -> dict:
    print("\n=== LLM judge: do the reasons match the tender? ===")
    judged = results.dropna(subset=["judge_label"])
    report = {"judged": int(len(judged)), "judge_match_rate": round(float(judged["judge_label"].mean()), 4) if len(judged) else None}
    print(f"  Judge: {report['judged']} cases judged" + (f", {report['judge_match_rate']:.0%} labelled as matching." if len(judged) else "."))
    labels = {r["id"].strip(): r["label"].strip().lower() for r in read_rows(HUMAN_LABELS)}
    human = {case_id: {"1": 1, "0": 0, "yes": 1, "no": 0, "match": 1, "mismatch": 0}.get(label) for case_id, label in labels.items()}
    both = judged[judged["id"].map(human).notna()]
    if len(both) < 2:
        print(f"  eval/human_labels.csv has {len(both)} labels that match a judged case: the team writes 20+ (id, label 1/0, comment).")
        return report | {"human_labels": int(len(both))}
    from sklearn.metrics import cohen_kappa_score

    mine, theirs = both["id"].map(human).astype(int), both["judge_label"].astype(int)
    kappa = cohen_kappa_score(mine, theirs) if mine.nunique() + theirs.nunique() > 2 else float("nan")
    report |= {"human_labels": int(len(both)), "agreement": round(float((mine == theirs).mean()), 4),
               "cohen_kappa": None if np.isnan(kappa) else round(float(kappa), 4)}
    print(f"  Against {len(both)} human labels: agreement {report['agreement']:.0%}, Cohen's kappa {report['cohen_kappa']}.")
    return report


def main() -> None:
    env.load_env()
    metrics = {"model": evaluate_model(), "agent": evaluate_agent(), "end_to_end": evaluate_cases(),
               "versions": {"agent_model": config.GEMINI_MODEL, "prompt_version": config.PROMPT_VERSION,
                            "model_version": config.MODEL_VERSION, "judge_version": judge.JUDGE_VERSION}}
    METRICS.write_text(json.dumps(metrics, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {METRICS.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
