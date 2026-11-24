"""
BidSense: should we bid for this public tender?

The two components and how one feeds the other:

    tender text
      -> redact personal data                      (bidsense/redact.py)
      -> component 1, Gemini agent                 (bidsense/agent.py): the tender's fields,
         each with a confidence and an evidence quote; tools: lookup_cpv, parse_amount, market_stats
      -> component 2, scikit-learn model           (bidsense/model.py): expected number of bidders,
         from the agent's fields only
      -> decision and review rule                  (bidsense/decision.py): BID / DON'T BID / SEND TO REVIEW

`run()` keeps the template's return format, because app.py, evaluate.py, main.py and
check.py all read it:

    run(raw_input) -> {
        "input_summary": str,          # short, no personal data: it is logged
        "features": {name: number},    # what the drift check watches
        "outputs": {component: ...},   # each component's output, JSON-serialisable
        "versions": {component: str},  # model and prompt versions
        "review": bool,                # does this case go to a person?
        "tokens": int,                 # tokens spent on this request (0 if none)
        "cost_eur": float,             # cost of this request in euros
    }

plus a few fields of our own (decision, tokens_in, tokens_out, language, cached).
"""

import math
from pathlib import Path

import config
from bidsense import agent, bandit, decision, lang, model
from bidsense.redact import redact

DECISION_RULE_VERSION = "ev-rule-v1"
VALUE_BUCKETS = [(100_000, "<€100k"), (500_000, "€100k-500k"), (1_000_000, "€500k-1M"), (5_000_000, "€1M-5M"), (math.inf, ">€5M")]


def read_input(raw_input: str) -> str:
    """Text as it is; for a file path, the file's text."""
    path = Path(str(raw_input))
    if len(str(raw_input)) < 300 and path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return str(raw_input)


def value_bucket(value) -> str:
    if value is None:
        return "value unknown"
    return next(label for limit, label in VALUE_BUCKETS if value < limit)


def input_features(text: str, language: str, values: dict, confidence: float) -> dict[str, float]:
    """Numbers that describe the request, for the drift check. 0 stands for "unknown" (never NaN: logs are JSON)."""
    value = values.get("estimated_value_eur")
    return {
        "is_spanish": 1.0 if language == "es" else 0.0,
        "cpv_division": float(values.get("cpv_division") or 0),
        "log_value": round(math.log10(value), 3) if value and value >= 1 else 0.0,
        "text_length": float(len(text)),
        "confidence": float(confidence),
    }


def run(raw_input: str, bid_cost_eur: float | None = None, margin_pct: float | None = None,
        prompt_version: str | None = None) -> dict:
    bid_cost_eur = config.DEFAULT_BID_COST_EUR if bid_cost_eur is None else float(bid_cost_eur)
    margin_pct = config.DEFAULT_MARGIN_PCT if margin_pct is None else float(margin_pct)
    prompt_version = prompt_version or bandit.choose()

    # Personal data is removed before anything else sees the text: Gemini, the logs, the cache.
    text, removed = redact(read_input(raw_input))
    language = lang.detect(text)

    values, confidence, prediction, baseline = {}, 0.0, None, None
    tokens_in = tokens_out = 0
    try:
        extraction = agent.extract(text, prompt_version)
    except agent.AgentError as error:
        agent_output = {"error": str(error)}
        outcome = {"decision": decision.REVIEW, "reason": f"The agent could not read this tender, so a person should decide. ({error})",
                   "win_chance": None, "expected_value_eur": None, "underlying": None, "review_reasons": ["agent unavailable"]}
    else:
        tokens_in, tokens_out = extraction["tokens_in"], extraction["tokens_out"]
        agent_output = {k: extraction[k] for k in ("is_tender", "fields", "market_note", "not_tender_reason", "steps", "cached", "latency_s")}
        values = {name: field["value"] for name, field in extraction["fields"].items()}
        confidence = min(extraction["fields"][name]["confidence"] for name in config.KEY_FIELDS)
        if not extraction["is_tender"]:
            # Out of scope: no prediction, no bid / no-bid.
            outcome = decision.out_of_scope(extraction["not_tender_reason"])
        else:
            # Component 1's output is component 2's input.
            prediction = model.predict(values)
            outcome = decision.decide(values, confidence, prediction["expected_bidders"], bid_cost_eur, margin_pct)
            # What the baseline (median bidders of the CPV division) would have decided, for the evaluation.
            base = decision.decide(values, confidence, prediction["baseline_bidders"], bid_cost_eur, margin_pct)
            baseline = {"expected_bidders": prediction["baseline_bidders"], "decision": base["decision"],
                        "underlying": base["underlying"], "expected_value_eur": base["expected_value_eur"]}

    division = values.get("cpv_division")
    return {
        "input_summary": f"cpv {division or '??'} | {value_bucket(values.get('estimated_value_eur'))} | {language} | {len(text):,} chars",
        "features": input_features(text, language, values, confidence),
        "outputs": {"agent": agent_output, "model": prediction, "decision": outcome, "baseline": baseline},
        "versions": {"agent_model": config.GEMINI_MODEL, "prompt_version": prompt_version,
                     "model_version": config.MODEL_VERSION, "decision_rule": DECISION_RULE_VERSION},
        "review": outcome["decision"] == decision.REVIEW,
        "tokens": int(tokens_in + tokens_out),
        "cost_eur": agent.cost_eur(tokens_in, tokens_out),
        # Our own fields (logbook.py keeps whatever is added):
        "decision": outcome["decision"],
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "language": language,
        "redactions": removed,
        "cached": bool(agent_output.get("cached", False)),
    }
