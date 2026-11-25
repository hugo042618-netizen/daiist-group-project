"""
LLM judge: does the app's reasoning match the tender?

The recommendation comes with reasons in free text and with the fields it relies on.
The judge (Gemini) reads the redacted tender text and the app's answer and says 1 if the
reasons are supported by the text and follow from it, 0 if not. Its labels are compared
with the team's own labels in eval/human_labels.csv (evaluate.py, Cohen's kappa).

Every call is cached on disk. Limitation to keep in mind: the judge is the same model
family as the agent, so it may share its blind spots.

Two rubrics. judge-v1 is the judge of the evaluation: it was written before the team labelled
anything. judge-v2 was written on 25 Nov 2026 AFTER comparing judge-v1 with the team's 31 labels,
to fix the two ways they disagreed (the judge took the bid's "expected value" for the contract's
value, and it accepted extracted facts that the text does not support). It is tuned on the same
labels it is then compared with, so its agreement is optimistic; it is reported next to judge-v1,
never instead of it. Both see the same text and the same answer.
"""

import json

import config
from bidsense import agent, cache

JUDGE_VERSION = "judge-v1"
RUBRIC = """You check the output of a bid / no-bid assistant for Spanish public tenders.

You get the tender text (personal data already removed) and the assistant's answer: its recommendation, its
reason, and the facts it extracted. Decide whether the reasons MATCH the tender:

label 1 (match) when all of these hold:
- every fact the reason relies on (contract value, what is bought, type of contract, whether it is a tender
  at all) is supported by the text, or is a sensible reading of it;
- the recommendation follows from the stated reason (for example a review because a key field is missing,
  when that field really is missing from the text);
- if the assistant says the text is not a tender, that is true.

label 0 (mismatch) when a fact is wrong or invented, a field is reported missing although the text states it
clearly, or the text is misclassified as a tender / not a tender.

Do NOT judge whether bidding is a good business decision, and do not judge the predicted number of bidders:
you cannot verify those from the text. Judge only whether the reasons match the tender.

Answer with JSON: {"label": 0 or 1, "explanation": "one sentence"}."""

RUBRIC_V2 = """You check the output of a bid / no-bid assistant for Spanish public tenders.

You get the tender text (personal data already removed) and the assistant's answer: its recommendation, its
reason, and the facts it extracted ("fields"). Decide whether the answer MATCHES the tender.

How to read the assistant's reason:
- "expected value" is the expected PROFIT of bidding: chance of winning x contract value x margin - cost of
  preparing the bid. It is a small number, often close to zero or negative. It is NOT the contract's value and
  must never be compared with the estimated value in the text. You cannot verify it, the number of expected
  bidders or the chance of winning from the text: never give label 0 because of them.
- The amount the reason calls the contract value ("winning EUR ...") is the fact to check against the text.

Check EVERY extracted fact in "fields", one by one, against the text:
- estimated_value_eur: the contract's estimated value without VAT. If the text gives an amount with VAT and one
  without, it must be the one without.
- duration_months: the text must state a duration, or dates that give it without guessing.
- procedure, lots_number, is_framework_agreement, region, contract_type, buyer_type: the text must state it or
  make it plain (the buyer's name for buyer_type, the place of performance for region).
- price_weight_pct: the text must give award criteria with the weight of price. If it gives no award criteria
  or no weights, any number here is a guess.
- cpv_code: the text may not print a code; then the code must fit what is being bought (check its meaning
  against the object of the contract, not against a number in the text). cpv_division follows the code.
- A fact with no value (null) is correct only if the text really does not give it.

label 1 (match) only when all of these hold:
- every extracted fact that has a value has a basis in the text, and no fact is reported missing although the
  text states it clearly;
- the recommendation follows from the stated reason (for example a review because the expected value is close
  to zero, because the contract is very large, or because a key field is missing);
- if the assistant says the text is not a tender, that is true (then there are no facts to check).

label 0 (mismatch) when any extracted fact is wrong, invented or guessed (a value the text gives no basis for),
when a fact is reported missing although the text states it, or when the text is misclassified as a tender / not
a tender. One such fact is enough.

Do NOT judge whether bidding is a good business decision.

Answer with JSON: {"label": 0 or 1, "explanation": "one sentence naming the fact that fails, if any"}."""

RUBRICS = {"judge-v1": RUBRIC, "judge-v2": RUBRIC_V2}
TUNED_VERSION = "judge-v2"   # written after seeing the team's labels: evaluation only, reported as optimistic


def judge(text: str, answer: dict, client=None, default_model_only: bool = False, version: str = JUDGE_VERSION) -> dict:
    """{label: 0|1, explanation, tokens_in, tokens_out, cached} for one case. `text` must be redacted.
    `version` chooses the rubric: "judge-v1" (the evaluation's judge) or "judge-v2" (tuned on the labels)."""
    text = text[: config.MAX_INPUT_CHARS]
    payload = json.dumps(answer, ensure_ascii=False, sort_keys=True)
    cache_key = cache.key(config.GEMINI_MODEL, version, text, payload)
    if client is None and (hit := cache.get("judge", cache_key)):
        return hit | {"cached": True}

    from google.genai import types

    client = client or agent._client()
    request_config = types.GenerateContentConfig(
        system_instruction=RUBRICS[version], temperature=0, response_mime_type="application/json",
        response_json_schema={"type": "object", "properties": {"label": {"type": "integer", "enum": [0, 1]},
                                                               "explanation": {"type": "string"}},
                              "required": ["label", "explanation"]},
        thinking_config=types.ThinkingConfig(thinking_level=config.GEMINI_THINKING),
    )
    contents = f"<tender_text>\n{text}\n</tender_text>\n\n<assistant_answer>\n{payload}\n</assistant_answer>"
    response = unavailable = None
    for model in agent.models_to_try(default_model_only):
        try:
            response = agent.call_gemini(client, model, contents, request_config)
            break
        except agent._Unavailable as error:
            unavailable = error
    if response is None:
        if not agent.models_to_try(default_model_only):
            raise agent.DailyQuotaExhausted("The free Gemini requests for today are used up.")
        raise agent.AgentError(str(unavailable or "no Gemini model is available"))
    try:
        parsed = json.loads(response.text)
        label = int(parsed["label"])
    except (TypeError, ValueError, KeyError) as error:
        raise agent.AgentError(f"The judge did not return valid JSON: {str(response.text)[:120]}") from error
    usage = response.usage_metadata
    result = {
        "label": 1 if label == 1 else 0,
        "explanation": str(parsed.get("explanation", ""))[:300],
        "tokens_in": int((usage.prompt_token_count or 0) if usage else 0),
        "tokens_out": int(((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0),
        "judge_version": version,
        "model": model,
    }
    cache.put("judge", cache_key, result)
    return result | {"cached": False}
