"""
The decision and the review rule. Every threshold lives in config.py (TEAM TO SET).

    win_chance     = min(1, 1 / expected_bidders)
    expected_value = win_chance * contract_value * margin - bid_cost

BID if the expected value is positive, DON'T BID if it is negative, and SEND TO REVIEW
(a person decides) when the contract is very large, a key field is missing, the agent
is not confident, or the expected value is too close to zero to call.
"""

import config

BID, DONT_BID, REVIEW, OUT_OF_SCOPE = "BID", "DONT_BID", "REVIEW", "OUT_OF_SCOPE"
LABELS = {BID: "BID", DONT_BID: "DON'T BID", REVIEW: "SEND TO REVIEW", OUT_OF_SCOPE: "OUT OF SCOPE"}


def _euros(amount: float) -> str:
    return f"€{amount:,.0f}"


def economics(expected_bidders: float, contract_value: float, bid_cost_eur: float, margin_pct: float) -> dict:
    win_chance = min(1.0, 1.0 / max(expected_bidders, 1e-9))
    expected_value = win_chance * contract_value * margin_pct / 100 - bid_cost_eur
    return {"win_chance": round(win_chance, 4), "expected_value_eur": round(expected_value, 2)}


def decide(fields: dict, confidence: float, expected_bidders: float | None,
           bid_cost_eur: float, margin_pct: float) -> dict:
    """The recommendation for one tender.

    `fields` are the agent's values ({name: value or None}); `confidence` is the lowest
    confidence among the key fields; `expected_bidders` comes from the model.
    """
    value = fields.get("estimated_value_eur")
    missing = [name for name in config.KEY_FIELDS if fields.get(name) in (None, "")]
    reasons = []
    if missing:
        reasons.append("key field missing: " + ", ".join(missing))
    if confidence < config.MIN_CONFIDENCE:
        reasons.append(f"the agent's confidence is {confidence:.2f}, below {config.MIN_CONFIDENCE}")
    if value is not None and value > config.REVIEW_VALUE_EUR:
        reasons.append(f"contract value {_euros(value)} is above {_euros(config.REVIEW_VALUE_EUR)}")

    result = {"win_chance": None, "expected_value_eur": None, "underlying": None,
              "bid_cost_eur": bid_cost_eur, "margin_pct": margin_pct}
    if value is not None and expected_bidders is not None:
        result |= economics(expected_bidders, value, bid_cost_eur, margin_pct)
        expected_value = result["expected_value_eur"]
        result["underlying"] = BID if expected_value > 0 else DONT_BID
        if abs(expected_value) < config.CLOSE_CALL_EUR:
            reasons.append(f"expected value {_euros(expected_value)} is within {_euros(config.CLOSE_CALL_EUR)} of zero: too close to call")

    if reasons:
        result |= {"decision": REVIEW, "reason": "A person should decide: " + "; ".join(reasons) + "."}
    else:
        expected_value = result["expected_value_eur"]
        sign = "positive" if expected_value > 0 else "negative"
        result |= {
            "decision": result["underlying"],
            "reason": (f"About {expected_bidders:.1f} bidders expected, so a {result['win_chance']:.0%} chance of winning "
                       f"{_euros(value)} at a {margin_pct:g}% margin against a bid cost of {_euros(bid_cost_eur)}: "
                       f"expected value {_euros(expected_value)} ({sign})."),
        }
    result["review_reasons"] = reasons
    return result


def out_of_scope(why: str) -> dict:
    return {"decision": OUT_OF_SCOPE, "reason": why or "This text is not a public tender, so there is nothing to bid on.",
            "win_chance": None, "expected_value_eur": None, "underlying": None, "review_reasons": []}
