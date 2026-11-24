"""
LLM judge: does the app's reasoning match the tender?

The recommendation comes with reasons in free text and with the fields it relies on.
The judge (Gemini) reads the redacted tender text and the app's answer and says 1 if the
reasons are supported by the text and follow from it, 0 if not. Its labels are compared
with the team's own labels in eval/human_labels.csv (evaluate.py, Cohen's kappa).

Every call is cached on disk. Limitation to keep in mind: the judge is the same model
family as the agent, so it may share its blind spots.
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


def judge(text: str, answer: dict, client=None) -> dict:
    """{label: 0|1, explanation, tokens_in, tokens_out, cached} for one case. `text` must be redacted."""
    text = text[: config.MAX_INPUT_CHARS]
    payload = json.dumps(answer, ensure_ascii=False, sort_keys=True)
    cache_key = cache.key(config.GEMINI_MODEL, JUDGE_VERSION, text, payload)
    if client is None and (hit := cache.get("judge", cache_key)):
        return hit | {"cached": True}

    from google.genai import types

    client = client or agent._client()
    request_config = types.GenerateContentConfig(
        system_instruction=RUBRIC, temperature=0, response_mime_type="application/json",
        response_json_schema={"type": "object", "properties": {"label": {"type": "integer", "enum": [0, 1]},
                                                               "explanation": {"type": "string"}},
                              "required": ["label", "explanation"]},
        thinking_config=types.ThinkingConfig(thinking_level=config.GEMINI_THINKING),
    )
    contents = f"<tender_text>\n{text}\n</tender_text>\n\n<assistant_answer>\n{payload}\n</assistant_answer>"
    response = unavailable = None
    for model in agent.models_to_try():
        try:
            response = agent.call_gemini(client, model, contents, request_config)
            break
        except agent._Unavailable as error:
            unavailable = error
    if response is None:
        if not agent.models_to_try():
            raise agent.DailyQuotaExhausted("Every Gemini model has used up its free requests for today.")
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
        "judge_version": JUDGE_VERSION,
        "model": model,
    }
    cache.put("judge", cache_key, result)
    return result | {"cached": False}
