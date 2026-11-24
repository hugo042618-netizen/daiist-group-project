"""
Component 1: the Gemini agent that reads a tender and extracts the fields.

`extract(text)` expects text that is ALREADY redacted (bidsense/redact.py). It runs a
function-calling loop: Gemini may call lookup_cpv, parse_amount and market_stats, and it
finishes by calling submit_extraction with the fields, each with a confidence and an
evidence quote. Every tool call is recorded in `steps`, and every run is cached on disk.

The code then checks what came back: codes must exist in the CPV list, enums must be
valid, and an evidence quote that is not in the text lowers that field's confidence.
"""

import os
import re
import time

import config
from bidsense import cache, features, prompts, tools
from bidsense.redact import scrub

TOOLS = {"lookup_cpv": tools.lookup_cpv, "parse_amount": tools.parse_amount, "market_stats": tools.market_stats}
SUBMIT = "submit_extraction"
UNVERIFIED_EVIDENCE_CAP = 0.5   # a quote that is not in the text cannot be trusted


class AgentError(RuntimeError):
    """The agent could not produce an extraction (missing key, quota, network, bad answer)."""


def _client():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise AgentError("GEMINI_API_KEY is not set. Put it in .env (see SETUP.md) or in the service's environment on Render.")
    from google import genai
    from google.genai import types

    # One attempt per call and a time limit: retries and the switch to a fallback model are handled here.
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=config.GEMINI_TIMEOUT_S * 1000, retry_options=types.HttpRetryOptions(attempts=1)))


class _Unavailable(AgentError):
    """This model cannot answer right now (overloaded or rate-limited): try the next one."""


_skip_until: dict[str, float] = {}   # model -> time until which it is skipped after being unavailable


def models_to_try() -> list[str]:
    """The configured model first, then the fallbacks; models that just failed are tried last."""
    models = [config.GEMINI_MODEL, *config.GEMINI_FALLBACK_MODELS]
    now = time.time()
    return sorted(models, key=lambda m: _skip_until.get(m, 0) > now)


def call_gemini(client, model: str, contents, request_config):
    """One call to one Gemini model, retried a little on rate limits and server errors."""
    from google.genai import errors

    for attempt in range(config.GEMINI_RETRIES + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=request_config)
        except errors.APIError as error:
            code = getattr(error, "code", None)
            message = f"Gemini request failed ({code}, {model}): {str(error)[:200]}"
            if code not in (429, 500, 502, 503, 504):
                raise AgentError(message) from error
            if attempt == config.GEMINI_RETRIES:
                _skip_until[model] = time.time() + config.MODEL_COOLDOWN_S
                raise _Unavailable(message) from error
            time.sleep((20 if code == 429 else 4) * (attempt + 1))
        except Exception as error:  # noqa: BLE001 - network problems and the like
            if attempt == config.GEMINI_RETRIES:
                raise _Unavailable(f"Gemini request failed ({model}): {type(error).__name__}: {str(error)[:200]}") from error
            time.sleep(4 * (attempt + 1))


def _request_config(system_prompt: str, allowed: list[str]):
    from google.genai import types

    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0,
        tools=[types.Tool(function_declarations=[types.FunctionDeclaration(**d) for d in prompts.TOOL_DECLARATIONS])],
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY", allowed_function_names=allowed)),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=types.ThinkingConfig(thinking_level=config.GEMINI_THINKING),
    )


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _clean_value(name: str, value):
    """A valid value for the field, or None."""
    if value is None or value == "":
        return None
    try:
        if name in ("is_tender", "is_framework_agreement"):
            return bool(value) if not isinstance(value, str) else value.strip().lower() in ("true", "yes", "sí", "si", "1")
        if name == "cpv_code":
            digits = re.sub(r"\D", "", str(value))[:8]
            return digits if digits in tools.cpv_codes() else None
        if name == "cpv_division":
            return features.cpv_division(re.sub(r"\D", "", str(value)).zfill(2))
        if name == "contract_type":
            return value if value in features.CONTRACT_TYPES else None
        if name == "procedure":
            return value if value in features.PROCEDURES else None
        if name == "buyer_type":
            return value if value in features.BUYER_TYPES else None
        if name == "estimated_value_eur":
            return float(value) if float(value) > 0 else None
        if name == "duration_months":
            return round(float(value), 1) if 0 < float(value) <= 600 else None
        if name == "lots_number":
            return int(value) if 1 <= int(value) <= 5000 else None
        if name == "price_weight_pct":
            return round(float(value), 1) if 0 <= float(value) <= 100 else None
        if name == "region":
            value = str(value).strip().upper()
            return value if value in features.REGIONS or value == "ES" else ("other" if value == "OTHER" else None)
        if name == "language":
            return str(value).strip().lower()[:2] or None
    except (TypeError, ValueError):
        return None
    return value


def _validate(submitted: dict, text: str) -> dict:
    """The agent's fields after the code's own checks: {name: {value, confidence, evidence, evidence_found}}."""
    haystack = _squash(text)
    fields = {}
    for name in prompts.FIELDS:
        raw = submitted.get(name) if isinstance(submitted.get(name), dict) else {}
        value = _clean_value(name, raw.get("value"))
        try:
            confidence = min(1.0, max(0.0, float(raw.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0.0
        evidence = str(raw.get("evidence") or "").strip()[:200]
        found = bool(evidence) and _squash(evidence).strip(" .\"'…") in haystack
        if evidence and not found:
            confidence = min(confidence, UNVERIFIED_EVIDENCE_CAP)
        if value is None:
            confidence = 0.0
        fields[name] = {"value": value, "confidence": round(confidence, 2), "evidence": evidence,
                        "evidence_found": found if evidence else None}
    # The division always follows the code, whatever the agent wrote.
    if fields["cpv_code"]["value"]:
        fields["cpv_division"] |= {"value": fields["cpv_code"]["value"][:2], "confidence": fields["cpv_code"]["confidence"]}
    return fields


def _short(value, limit: int = 600):
    text = str(value)
    return value if len(text) <= limit else text[:limit] + "…"


def _run(client, model: str, text: str, system_prompt: str) -> tuple[dict, list, int, int]:
    """One conversation with one model: tool calls until submit_extraction. Returns (submitted, steps, tokens in, tokens out)."""
    from google.genai import types

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=f"<tender_text>\n{text}\n</tender_text>")])]
    steps, tokens_in, tokens_out = [], 0, 0
    for turn in range(config.MAX_AGENT_STEPS):
        last_turn = turn == config.MAX_AGENT_STEPS - 1
        response = call_gemini(client, model, contents, _request_config(system_prompt, [SUBMIT] if last_turn else [*TOOLS, SUBMIT]))
        usage = response.usage_metadata
        if usage is not None:
            tokens_in += (usage.prompt_token_count or 0) + (getattr(usage, "tool_use_prompt_token_count", None) or 0)
            tokens_out += (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        calls = response.function_calls or []
        if not calls:
            raise AgentError("Gemini answered without calling a tool.")
        contents.append(response.candidates[0].content)   # keeps the model's own turn, thought signatures included
        replies = []
        for call in calls:
            arguments = dict(call.args or {})
            if call.name == SUBMIT:
                return arguments, steps, tokens_in, tokens_out
            if call.name in TOOLS:
                try:
                    result = TOOLS[call.name](**arguments)
                except Exception as error:  # noqa: BLE001 - a bad tool call is an answer for the model, not a crash
                    result = {"error": f"{type(error).__name__}: {error}"}
            else:
                result = {"error": f"unknown tool {call.name}"}
            steps.append({"tool": call.name, "args": arguments, "result": _short(result)})
            replies.append(types.Part.from_function_response(name=call.name, response=result))
        contents.append(types.Content(role="user", parts=replies))
    raise AgentError(f"Gemini did not submit an extraction within {config.MAX_AGENT_STEPS} steps.")


def extract(text: str, prompt_version: str | None = None, client=None) -> dict:
    """Run the agent on a (redacted) tender text.

    Returns {fields, is_tender, not_tender_reason, market_note, steps, tokens_in,
    tokens_out, latency_s, model, prompt_version, cached}. Raises AgentError if it cannot.
    If the configured model is unavailable, the whole run is done again with a fallback model.
    """
    prompt_version = prompt_version or config.PROMPT_VERSION
    text = text[: config.MAX_INPUT_CHARS]
    cache_key = cache.key(config.GEMINI_MODEL, config.GEMINI_THINKING, prompt_version, text)
    if client is None and (hit := cache.get("agent", cache_key)):
        return hit | {"cached": True}

    client = client or _client()
    started = time.perf_counter()
    unavailable = None
    for model in models_to_try():
        try:
            submitted, steps, tokens_in, tokens_out = _run(client, model, text, prompts.PROMPTS[prompt_version])
            break
        except _Unavailable as error:
            unavailable = error
    else:
        raise AgentError(str(unavailable))

    fields = _validate(submitted, text)
    # What the model wrote is filtered again before it is cached or logged: evidence quotes are checked
    # against the text first (above), then every free-text output goes through the strict name filter.
    for field in fields.values():
        field["evidence"] = scrub(field["evidence"])
    result = {
        "fields": fields,
        "is_tender": bool(fields["is_tender"]["value"]),
        "not_tender_reason": scrub(str(submitted.get("not_tender_reason") or "")[:300]),
        "market_note": scrub(str(submitted.get("market_note") or "")[:300]),
        "steps": scrub(steps),
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "latency_s": round(time.perf_counter() - started, 3),
        "model": model,
        "prompt_version": prompt_version,
    }
    cache.put("agent", cache_key, result)
    return result | {"cached": False}


def cost_eur(tokens_in: int, tokens_out: int, model: str | None = None) -> float:
    """What these tokens cost at Gemini's paid-tier prices for that model (config.py), in euros."""
    price_in, price_out = config.PRICES_USD_PER_M.get(model or config.GEMINI_MODEL, config.PRICES_USD_PER_M[config.GEMINI_MODEL])
    return round((tokens_in / 1e6 * price_in + tokens_out / 1e6 * price_out) * config.USD_TO_EUR, 6)
