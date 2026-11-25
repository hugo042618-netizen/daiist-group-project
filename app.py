"""
The app: an App tab and a Monitoring tab. `uv run python app.py`

The logic is in pipeline.py and bidsense/; this file only wires it to the interface and
the logs. `serve()` and the `/run` endpoint are kept from the template: check.py calls
`serve()` locally and `/run` on the deployed app, with one input from eval/cases.csv.
"""

import html
import json
import os
import time

import gradio as gr
import pandas as pd

import config
import env
import logbook
import monitor
import pipeline
from bidsense import bandit, decision, evidence, features, tools
from bidsense.redact import redact

env.load_env()
LOG_TARGET = logbook.start()

TITLE = "BidSense"
TAGLINE = "Know in seconds whether a Spanish public tender is worth bidding for."
HOW_IT_WORKS = ["Paste the tender", "AI reads it", "Model predicts the competition", "Bid, don't bid or review"]
EXAMPLES = json.loads((config.DATA_DIR / "examples.json").read_text(encoding="utf-8"))

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)
# Our own blocks use Gradio's colour variables for neutral surfaces, so they follow the light and the dark
# theme, and solid colours with white text for the decision, so it reads the same in both.
CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
.bs-header { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 10px 28px; padding: 6px 2px 2px; }
.bs-header h1 { font-size: 1.75rem; font-weight: 750; letter-spacing: -0.025em; margin: 0; line-height: 1.1; }
.bs-header p { margin: 5px 0 0; font-size: 1rem; color: var(--body-text-color-subdued); }
.bs-how { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; font-size: .84rem; color: var(--body-text-color-subdued); }
.bs-how span.bs-step { display: inline-flex; align-items: center; gap: 7px; }
.bs-how b { display: inline-grid; place-items: center; width: 20px; height: 20px; border-radius: 50%; font-size: .72rem; font-weight: 700;
  background: var(--color-accent-soft); color: var(--color-accent); }
.bs-how i { font-style: normal; opacity: .5; }
.bs-kicker { font-size: .72rem; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; color: var(--body-text-color-subdued); margin: 4px 2px -12px; }
.bs-chips { gap: 8px !important; flex-wrap: wrap !important; justify-content: flex-start !important; }
.bs-chips > * { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
button.bs-chip { border-radius: 999px !important; padding: 6px 15px !important; font-weight: 550 !important; font-size: .85rem !important;
  border: 1px solid var(--border-color-primary) !important; background: var(--background-fill-primary) !important; }
button.bs-chip:hover { border-color: var(--color-accent) !important; color: var(--color-accent) !important; }
.gradio-container button { transition: transform 140ms cubic-bezier(0.23, 1, 0.32, 1), border-color 140ms ease, color 140ms ease; }
.gradio-container button:active { transform: scale(0.97); }

.bs-hero { border-radius: 14px; padding: 24px 28px 26px; color: #fff; animation: bs-in 220ms cubic-bezier(0.23, 1, 0.32, 1); }
.bs-hero .bs-label { font-size: .74rem; font-weight: 650; letter-spacing: .1em; text-transform: uppercase; color: rgba(255,255,255,.82); }
.bs-hero .bs-decision { font-size: 3.1rem; font-weight: 800; letter-spacing: -0.035em; line-height: 1.05; margin: 4px 0 12px; color: #fff; }
.bs-hero .bs-reason { font-size: 1rem; line-height: 1.5; color: rgba(255,255,255,.95); max-width: 62ch; }
.bs-hero ul { margin: 6px 0 0 18px; padding: 0; }
.bs-hero li { color: rgba(255,255,255,.95); }
.bs-bid { background: #0b8a5f; }
.bs-dont { background: #c62828; }
.bs-review { background: #b45309; }
.bs-neutral { background: #475569; }
@keyframes bs-in { from { opacity: 0; transform: translateY(6px) scale(.985); } to { opacity: 1; transform: none; } }

.bs-placeholder { border: 1.5px dashed var(--border-color-primary); border-radius: 14px; padding: 34px 30px; background: var(--background-fill-secondary); }
.bs-placeholder .bs-ph-title { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.015em; margin: 12px 0 4px; color: var(--body-text-color); }
.bs-placeholder .bs-ph-text { color: var(--body-text-color-subdued); margin-bottom: 14px; }
.bs-placeholder ul { margin: 0; padding: 0; list-style: none; display: grid; gap: 8px; }
.bs-placeholder li { display: flex; gap: 10px; align-items: baseline; color: var(--body-text-color); font-size: .94rem; }
.bs-placeholder li::before { content: ""; flex: 0 0 7px; height: 7px; border-radius: 50%; background: var(--color-accent); transform: translateY(-1px); }
.bs-ph-icon { width: 44px; height: 44px; border-radius: 12px; display: grid; place-items: center; background: var(--color-accent-soft); color: var(--color-accent); }

.bs-loading { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 26px 28px; background: var(--background-fill-secondary); }
.bs-loading .bs-label { font-size: .74rem; font-weight: 650; letter-spacing: .1em; text-transform: uppercase; color: var(--body-text-color-subdued); }
.bs-loading ol { list-style: none; margin: 12px 0 0; padding: 0; display: grid; gap: 10px; }
.bs-loading li { display: flex; align-items: center; gap: 12px; font-size: 1.12rem; font-weight: 600; color: var(--body-text-color); opacity: .32;
  animation: bs-step 400ms ease-out forwards; }
.bs-loading li::before { content: ""; width: 10px; height: 10px; border-radius: 50%; background: var(--color-accent); animation: bs-pulse 900ms ease-in-out infinite; }
.bs-loading li:nth-child(1) { animation-delay: 0ms; }
.bs-loading li:nth-child(2) { animation-delay: 1800ms; }
.bs-loading li:nth-child(3) { animation-delay: 3200ms; }
@keyframes bs-step { to { opacity: 1; } }
@keyframes bs-pulse { 0%, 100% { opacity: .35; transform: scale(.8); } 50% { opacity: 1; transform: scale(1); } }

.bs-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.bs-tiles.bs-three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.bs-tiles.bs-five { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.bs-tile { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 14px 16px; background: var(--background-fill-primary); min-width: 0; }
.bs-tile .bs-k { font-size: .7rem; font-weight: 650; letter-spacing: .07em; text-transform: uppercase; color: var(--body-text-color-subdued); }
.bs-tile .bs-v { font-size: 1.7rem; font-weight: 750; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; margin-top: 3px; color: var(--body-text-color); }
.bs-tile .bs-s { font-size: .8rem; color: var(--body-text-color-subdued); }

.bs-section { font-size: .95rem; font-weight: 700; margin: 6px 2px 8px; color: var(--body-text-color); }
.bs-section span { font-weight: 450; color: var(--body-text-color-subdued); margin-left: 6px; font-size: .84rem; }
.bs-table-wrap { border: 1px solid var(--border-color-primary); border-radius: 12px; overflow: hidden; background: var(--background-fill-primary); }
table.bs-table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: .9rem; margin: 0; border: 0 !important; }
table.bs-table th { text-align: left; font-size: .68rem; font-weight: 650; letter-spacing: .05em; text-transform: uppercase; padding: 10px 12px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--body-text-color-subdued); background: var(--background-fill-secondary);
  border: 0 !important; border-bottom: 1px solid var(--border-color-primary) !important; }
table.bs-table td { padding: 9px 12px; vertical-align: middle; color: var(--body-text-color); overflow-wrap: anywhere;
  border: 0 !important; border-bottom: 1px solid var(--border-color-primary) !important; }
table.bs-table tr:last-child td { border-bottom: 0 !important; }
table.bs-table thead, table.bs-table tbody, table.bs-table tr { border: 0 !important; background: transparent !important; }
table.bs-table td.bs-quiet { color: var(--body-text-color-subdued); font-size: .86rem; }
table.bs-table td.bs-f { font-weight: 600; }
table.bs-table td.bs-num { font-variant-numeric: tabular-nums; }
table.bs-table td .bs-sub { display: block; font-size: .78rem; color: var(--body-text-color-subdued); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
table.bs-table td.bs-ev { color: var(--body-text-color-subdued); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: help; }
.bs-pill { display: inline-block; min-width: 46px; text-align: center; border-radius: 999px; padding: 2px 9px; font-size: .8rem; font-weight: 650;
  font-variant-numeric: tabular-nums; }
.bs-good { background: #d7f5e6; color: #0a6b48; }
.bs-mid { background: #fdecc8; color: #8a4b00; }
.bs-bad { background: #fde0e0; color: #a01d1d; }
.bs-flat { background: var(--background-fill-secondary); color: var(--body-text-color-subdued); }
.bs-miss { display: inline-block; margin-right: 7px; border-radius: 5px; padding: 1px 6px; font-size: .68rem; font-weight: 650; letter-spacing: .02em;
  background: #fde0e0; color: #a01d1d; vertical-align: 1px; }

.bs-banner { border-radius: 12px; padding: 14px 18px; display: flex; gap: 14px; align-items: flex-start; color: #fff; }
.bs-banner .bs-dot { flex: 0 0 12px; height: 12px; border-radius: 50%; background: rgba(255,255,255,.9); margin-top: 6px; }
.bs-banner b { font-size: 1.08rem; font-weight: 700; display: block; color: #fff; }
.bs-banner div, .bs-banner li { font-size: .9rem; line-height: 1.5; color: rgba(255,255,255,.95); }
.bs-banner ul { margin: 4px 0 4px 18px; padding: 0; }
.bs-banner small { display: block; margin-top: 4px; font-size: .8rem; color: rgba(255,255,255,.8); }
.bs-empty { border: 1.5px dashed var(--border-color-primary); border-radius: 12px; padding: 26px 20px; text-align: center;
  background: var(--background-fill-secondary); color: var(--body-text-color); font-weight: 650; }
.bs-empty span { display: block; margin-top: 3px; font-weight: 400; font-size: .86rem; color: var(--body-text-color-subdued); }
.bs-foot { font-size: .8rem; color: var(--body-text-color-subdued); }
@media (max-width: 900px) { .bs-tiles.bs-five { grid-template-columns: repeat(2, minmax(0, 1fr)); } .bs-hero .bs-decision { font-size: 2.4rem; } }
@media (prefers-reduced-motion: reduce) {
  .bs-hero, .bs-loading li, .bs-loading li::before { animation: none; }
  .bs-loading li { opacity: 1; }
  .gradio-container button:active { transform: none; }
}
"""

FIELD_LABELS = {
    "cpv_code": "CPV code", "contract_type": "Contract type", "estimated_value_eur": "Estimated value",
    "duration_months": "Duration", "procedure": "Procedure", "lots_number": "Lots", "price_weight_pct": "Weight of price",
    "is_framework_agreement": "Framework agreement", "region": "Region", "buyer_type": "Buyer type", "language": "Language",
}
CARD_CLASS = {decision.BID: "bs-bid", decision.DONT_BID: "bs-dont", decision.REVIEW: "bs-review", decision.OUT_OF_SCOPE: "bs-neutral"}
EVIDENCE_CHARS = 64   # the table shows this much of a quote; the whole quote is the cell's tooltip
PLACEHOLDER_CARD = (
    '<div class="bs-placeholder"><div class="bs-ph-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'
    '<path d="M14 3v5h5"/><path d="M9 13h6"/><path d="M9 17h4"/></svg></div>'
    '<div class="bs-ph-title">Your recommendation will appear here</div>'
    '<div class="bs-ph-text">Paste a tender on the left, or try an example, and press Analyse.</div>'
    "<ul><li><span><b>Bid, don't bid or send to review</b>, with the reason</span></li>"
    "<li><span>How many bidders to expect, and your chance of winning</span></li>"
    "<li><span>Every fact the AI read, with the quote it came from</span></li></ul></div>")
# The three lines light up on a timer while the request runs; they are a progress hint, not a measurement.
LOADING_CARD = ('<div class="bs-loading"><div class="bs-label">Analysing</div><ol><li>Reading the tender…</li>'
                "<li>Estimating competition…</li><li>Deciding…</li></ol></div>")


def settings_title(bid_cost_eur=None, margin_pct=None) -> str:
    """The title of the company settings, with the values in use."""
    cost = config.DEFAULT_BID_COST_EUR if bid_cost_eur in (None, "") else float(bid_cost_eur)
    margin = config.DEFAULT_MARGIN_PCT if margin_pct in (None, "") else float(margin_pct)
    return f"Company settings · bid cost €{cost:,.0f} · margin {margin:g}%"


def serve(raw_input: str, bid_cost_eur: float | None = None, margin_pct: float | None = None) -> dict:
    """Run one request through the pipeline, log it, and return the logged record."""
    started = time.perf_counter()
    result = pipeline.run(raw_input, bid_cost_eur, margin_pct)
    return logbook.log_request(result | {"latency_s": round(time.perf_counter() - started, 3)})


def run_api(raw_input: str) -> dict:
    """The `/run` endpoint: one tender text in, the recommendation out (default company settings)."""
    record = serve(raw_input)
    outputs = record["outputs"]
    return {
        "id": record["id"], "decision": record["decision"], "reason": outputs["decision"]["reason"], "review": record["review"],
        "expected_bidders": (outputs["model"] or {}).get("expected_bidders"),
        "win_chance": outputs["decision"]["win_chance"], "expected_value_eur": outputs["decision"]["expected_value_eur"],
        "fields": {name: field["value"] for name, field in (outputs["agent"].get("fields") or {}).items()},
        "latency_s": record["latency_s"], "cost_eur": record["cost_eur"],
    }


# --- App tab -----------------------------------------------------------------

def _tile(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="bs-tile"><div class="bs-k">{html.escape(label)}</div><div class="bs-v">{html.escape(value)}</div>'
            f'<div class="bs-s">{html.escape(sub)}</div></div>')


def _card(outcome: dict) -> str:
    """The hero: the recommendation, big and coloured, with its reason."""
    reasons = outcome.get("review_reasons") or []
    if outcome["decision"] == decision.REVIEW and reasons:
        body = "A person should decide:<ul>" + "".join(f"<li>{html.escape(r)}</li>" for r in reasons) + "</ul>"
    else:
        body = html.escape(outcome["reason"])
    return (f'<div class="bs-hero {CARD_CLASS[outcome["decision"]]}"><div class="bs-label">Recommendation</div>'
            f'<div class="bs-decision">{decision.LABELS[outcome["decision"]]}</div><div class="bs-reason">{body}</div></div>')


def _tiles(prediction: dict | None, outcome: dict) -> str:
    if not prediction:
        return ('<div class="bs-tiles bs-three">' + _tile("Expected bidders", "–", "no prediction") + _tile("Win chance", "–")
                + _tile("Expected value", "–") + "</div>")
    win, value = outcome.get("win_chance"), outcome.get("expected_value_eur")
    return '<div class="bs-tiles bs-three">' + "".join([
        _tile("Expected bidders", f"{prediction['expected_bidders']:.1f}",
              f"80% range {prediction['range_low']:.0f} to {prediction['range_high']:.0f}"),
        _tile("Win chance", f"{win:.0%}" if win is not None else "–", "1 / expected bidders"),
        _tile("Expected value", f"€{value:,.0f}" if value is not None else "–",
              f"margin {outcome.get('margin_pct', 0):g}%, bid cost €{outcome.get('bid_cost_eur', 0):,.0f}" if value is not None else "value missing"),
    ]) + "</div>"


def _show(name: str, value) -> str:
    if value is None:
        return "not found"
    if name == "estimated_value_eur":
        return f"€{value:,.0f}"
    if name == "duration_months":
        return f"{value:g} months"
    if name == "price_weight_pct":
        return f"{value:g}%"
    if name == "region":
        return f"{value} ({features.REGIONS[value]})" if value in features.REGIONS else str(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).replace("_", " ")


def _confidence(field: dict) -> str:
    """The confidence as a percentage in a coloured pill; grey when there is no value."""
    if field.get("value") is None:
        return '<span class="bs-pill bs-flat">–</span>'
    confidence = float(field.get("confidence") or 0)
    level = "bs-good" if confidence >= 0.8 else "bs-mid" if confidence >= config.MIN_CONFIDENCE else "bs-bad"
    return f'<span class="bs-pill {level}">{confidence:.0%}</span>'


def _found(agent_output: dict, text: str | None = None) -> str:
    """"What the agent found": one row per field, the confidence as a percentage, and a short evidence quote
    whose full text shows on hover. With `text` (what the agent read), a quote that is not in it gets a small
    "not found in text" mark: for the reader only, it changes no confidence and no decision."""
    rows = []
    for name, label in FIELD_LABELS.items():
        field = (agent_output.get("fields") or {}).get(name)
        if field is None:
            continue
        value = html.escape(_show(name, field["value"]))
        if name == "cpv_code" and field["value"] in tools.cpv_codes():
            name_en = html.escape(tools.cpv_codes()[field["value"]]["en"], quote=True)
            value += f'<span class="bs-sub" title="{name_en}">{name_en}</span>'
        quote = str(field.get("evidence") or "").strip()
        missing = evidence.in_text(quote, text) is False if text is not None else field.get("evidence_found") is False
        mark = '<span class="bs-miss">not found in text</span>' if quote and missing else ""
        room = EVIDENCE_CHARS - (22 if mark else 0)
        short = quote if len(quote) <= room else quote[:room].rstrip() + "…"
        shown = f"{mark}“{html.escape(short)}”" if quote else '<span style="opacity:.6">inferred, no quote</span>' if field["value"] is not None else ""
        tip = quote + (" (this quote does not appear in the text, ignoring case and spaces)" if mark else "")
        rows.append(f'<tr><td class="bs-f">{html.escape(label)}</td><td class="bs-num">{value}</td><td>{_confidence(field)}</td>'
                    f'<td class="bs-ev" title="{html.escape(tip, quote=True)}">{shown}</td></tr>')
    if not rows:
        return '<div class="bs-section">What the agent found</div><div class="bs-empty">Nothing was extracted<span>The agent did not read any field from this text.</span></div>'
    return ('<div class="bs-section">What the agent found<span>hover a quote to read it in full</span></div>'
            '<div class="bs-table-wrap"><table class="bs-table"><colgroup><col style="width:20%"><col style="width:25%"><col style="width:17%">'
            '<col style="width:38%"></colgroup><thead><tr><th>Field</th><th>Value</th><th>Confidence</th><th>Evidence</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def _steps(record: dict) -> str:
    agent_output = record["outputs"]["agent"]
    if agent_output.get("error"):
        return f"The agent did not run: {agent_output['error']}"
    removed = record.get("redactions") or {}
    lines = ["**0. Redaction** (before anything is sent to Gemini): "
             + (", ".join(f"{n} {kind}" for kind, n in removed.items()) + " removed" if removed else "nothing to remove")]
    for i, step in enumerate(agent_output.get("steps") or [], 1):
        lines.append(f"**{i}. {step['tool']}**(`{json.dumps(step['args'], ensure_ascii=False)}`)\n\n```json\n"
                     f"{json.dumps(step['result'], ensure_ascii=False, indent=1)[:900]}\n```")
    lines.append(f"**{len(lines)}. submit_extraction** → the table above"
                 + (f"\n\n_{agent_output['market_note']}_" if agent_output.get("market_note") else ""))
    if record["outputs"]["model"]:
        prediction = record["outputs"]["model"]
        lines.append(f"**Model** ({prediction['model_version']}): {prediction['expected_bidders']} expected bidders; "
                     f"baseline for this CPV division: {prediction['baseline_bidders']:g}.")
    return "\n\n".join(lines)


def on_analyse(raw_input, bid_cost_eur, margin_pct):
    """First the progress card, then the result. Outputs: card, details column, tiles, found, steps, footer, request id, comment."""
    if raw_input is None or not str(raw_input).strip():
        gr.Warning("Paste the text of a tender first.")
        yield [gr.update() for _ in range(8)]
        return
    yield [LOADING_CARD, gr.update(visible=False)] + [gr.update() for _ in range(6)]
    record = serve(raw_input, bid_cost_eur, margin_pct)
    outputs = record["outputs"]
    footer = (f"Latency {record['latency_s']} s · {record['tokens_in']:,} tokens in, {record['tokens_out']:,} out · "
              f"estimated cost €{record['cost_eur']:.4f} (paid-tier prices)"
              + (" · answer served from the cache" if record.get("cached") else "")
              + f" · model {record['versions']['model_version']}, prompt {record['versions']['prompt_version']}")
    # An out-of-scope text has no prediction and no extracted facts worth a confidence: only the card is shown.
    in_scope = record["decision"] != decision.OUT_OF_SCOPE
    read = redact(pipeline.read_input(raw_input))[0][: config.MAX_INPUT_CHARS]   # what the agent read, for the evidence marks
    yield [_card(outputs["decision"]), gr.update(visible=True),
           gr.update(value=_tiles(outputs["model"], outputs["decision"]) if in_scope else "", visible=in_scope),
           gr.update(value=_found(outputs["agent"], read) if in_scope else "", visible=in_scope),
           _steps(record), footer, record["id"], ""]


def on_feedback(request_id, value, comment):
    if not request_id:
        gr.Warning("Analyse a tender first.")
        return gr.update()
    logbook.log_feedback(request_id, value, comment or "")
    gr.Info("Thanks for the feedback.")
    return ""


# --- Monitoring tab ----------------------------------------------------------

DRIFT_LABELS = {"language_in_scope": "Language in scope", "cpv_division": "CPV division (sector)", "log_value": "Contract value (log)",
                "text_length": "Text length", "texts in a language out of scope": "Texts in a language out of scope"}


def _percent(value) -> str:
    return "–" if value is None else f"{value:.0%}"


def _empty(why: str) -> str:
    return f'<div class="bs-empty">No data yet<span>{html.escape(why)}</span></div>'


def _banner(kind: str, title: str, body: str, rule: str = "") -> str:
    return (f'<div class="bs-banner {kind}"><span class="bs-dot"></span><div><b>{html.escape(title)}</b>{body}'
            + (f"<small>{html.escape(rule)}</small>" if rule else "") + "</div></div>")


def _safely(name: str, build, fallback):
    """One panel of the Monitoring tab. A panel that fails shows its empty state instead of an error, and the
    reason goes to the server log."""
    try:
        return build()
    except Exception as error:  # noqa: BLE001 - a broken panel must not take the others down
        print(f"[monitoring] {name}: {type(error).__name__}: {error}", flush=True)
        return fallback


def _kpi_tiles(logs: pd.DataFrame) -> str:
    k = monitor.kpis(logs)
    return '<div class="bs-tiles bs-five">' + "".join([
        _tile("Requests", f"{k['requests']:,}", "since the first request"),
        _tile("Sent to review", _percent(k["review_share"]), "of all requests"),
        _tile("Thumbs down", _percent(k["thumbs_down_share"]), f"of {k['rated']} rated"),
        _tile("Avg cost / request", "–" if k["mean_cost_eur"] is None else f"€{k['mean_cost_eur']:.4f}", "paid-tier estimate"),
        _tile("Avg latency", "–" if k["mean_latency_s"] is None else f"{k['mean_latency_s']:.1f} s", "per request"),
    ]) + "</div>"


def _drift_banner(logs: pd.DataFrame, reference) -> str:
    rule = (f"Alert when a feature's PSI is above {monitor.PSI_ALERT} and above {monitor.NOISE_MULTIPLE}× the PSI expected by chance, "
            f"when more than {monitor.OTHER_LANGUAGE_ALERT:.0%} of texts are not in Spanish, Catalan, Galician or Basque, or when more "
            f"than {monitor.THUMBS_DOWN_ALERT:.0%} of rated requests are 👎. Window: the last {monitor.WINDOW} requests.")
    if reference is None:
        return _banner("bs-neutral", "No reference set yet", "<div>Run <code>uv run python main.py reference</code> and commit "
                       "<code>monitoring/</code>.</div>")
    if len(logs) < monitor.MIN_WINDOW:
        return _banner("bs-neutral", "Drift check: waiting for data",
                       f"<div>{len(logs)} of the {monitor.MIN_WINDOW} requests it needs before it can say anything.</div>", rule)
    fired = monitor.alerts(logs, reference)
    if fired:
        return _banner("bs-dont", "Alert", "<ul>" + "".join(f"<li>{html.escape(m)}</li>" for m in fired) + "</ul>", rule)
    return _banner("bs-bid", "No drift", f"<div>The last {min(len(logs), monitor.WINDOW)} requests look like the reference set, "
                   "and the feedback is within its limit.</div>", rule)


def _drift_table(logs: pd.DataFrame, reference) -> str:
    if reference is None or len(logs) < monitor.MIN_WINDOW:
        return _empty(f"The drift table needs a reference set and at least {monitor.MIN_WINDOW} requests.")
    report = monitor.drift(reference, monitor.live_features(logs))
    if report.empty:
        return _empty("No drift feature could be computed from the logs.")
    rows = []
    for row in report.itertuples():
        value = "–" if pd.isna(row.value) else (f"{row.value:.0%}" if row.metric == "share" else f"{row.value:.3f}")
        limit = f"{row.threshold:.0%}" if row.metric == "share" else f"{row.threshold:.3f}"
        badge = '<span class="bs-pill bs-bad">alert</span>' if row.alert else '<span class="bs-pill bs-good">ok</span>'
        rows.append(f'<tr><td class="bs-f">{html.escape(DRIFT_LABELS.get(row.feature, row.feature))}</td><td>{html.escape(row.metric)}</td>'
                    f'<td class="bs-num">{value}</td><td class="bs-num">{limit}</td><td>{badge}</td></tr>')
    return ('<div class="bs-table-wrap"><table class="bs-table"><thead><tr><th>Feature</th><th>Measure</th><th>Value</th>'
            "<th>Alert above</th><th>Status</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


DECISION_PILL = {decision.BID: "bs-good", decision.DONT_BID: "bs-bad", decision.REVIEW: "bs-mid", decision.OUT_OF_SCOPE: "bs-flat"}


def _recent_table(logs: pd.DataFrame) -> str:
    """The latest requests: summaries only, never the tender text."""
    recent = monitor.recent(logs)
    if recent.empty:
        return _empty("Analysed tenders are listed here (summaries only, never the text).")
    rows = []
    for row in recent.to_dict("records"):
        word = str(row["decision"] or "")
        pill = f'<span class="bs-pill {DECISION_PILL.get(word, "bs-flat")}">{html.escape(decision.LABELS.get(word, word) or "–")}</span>'
        bidders = row["expected bidders"]
        confidence = pd.to_numeric(row["confidence"], errors="coerce")   # monitor.recent() leaves it empty for out-of-scope texts
        latency = pd.to_numeric(row["latency (s)"], errors="coerce")
        rows.append(
            f'<tr><td class="bs-quiet bs-num">{html.escape(str(row["time (UTC)"]))}</td><td>{html.escape(str(row["input summary"]))}</td>'
            f"<td>{pill}</td>"
            f'<td class="bs-num">{"–" if bidders in ("", None) or pd.isna(bidders) else f"{float(bidders):.1f}"}</td>'
            f'<td class="bs-num">{"–" if pd.isna(confidence) else f"{confidence:.0%}"}</td>'
            f'<td class="bs-num">{"–" if pd.isna(latency) else f"{latency:.1f} s"}</td>'
            f'<td class="bs-num">{"–" if row["cost (€)"] == "" else f"€{float(row['cost (€)']):.4f}"}</td>'
            f'<td>{html.escape(str(row["feedback"])) or "–"}</td></tr>')
    return ('<div class="bs-table-wrap"><table class="bs-table"><colgroup><col style="width:12%"><col style="width:31%"><col style="width:15%">'
            '<col style="width:9%"><col style="width:10%"><col style="width:8%"><col style="width:8%"><col style="width:7%"></colgroup>'
            "<thead><tr><th>Time (UTC)</th><th>Input summary</th><th>Decision</th><th>Bidders</th><th>Confidence</th><th>Latency</th>"
            "<th>Cost</th><th>Rated</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def on_refresh():
    """Every panel of the Monitoring tab. Each one is built on its own, so one failure cannot blank the page."""
    logs = _safely("logs", logbook.load, pd.DataFrame())
    reference = _safely("reference", monitor.load_reference, None)
    quality = _safely("quality", lambda: monitor.quality(logs), pd.DataFrame(columns=["time", "signal", "value"]))
    arms = _safely("bandit", lambda: pd.DataFrame([{"prompt": arm} | row for arm, row in bandit.state(logs).items()]), pd.DataFrame())
    return (
        _safely("kpis", lambda: _kpi_tiles(logs), _kpi_tiles(pd.DataFrame())),
        _safely("banner", lambda: _drift_banner(logs, reference), _banner("bs-neutral", "No data yet", "<div>The drift check could not run.</div>")),
        gr.update(value=quality, visible=not quality.empty),
        gr.update(value=_empty("The chart appears after the first request."), visible=quality.empty),
        _safely("drift table", lambda: _drift_table(logs, reference), _empty("The drift table could not be computed.")),
        _safely("recent", lambda: _recent_table(logs), _empty("The recent requests could not be listed.")),
        gr.update(value=arms, visible=not arms.empty),
        gr.update(value=_empty("No prompt has been tried yet."), visible=arms.empty),
    )


with gr.Blocks(title=TITLE) as demo:
    how = "<i>→</i>".join(f'<span class="bs-step"><b>{i}</b>{html.escape(step)}</span>' for i, step in enumerate(HOW_IT_WORKS, 1))
    gr.HTML(f'<div class="bs-header"><div><h1>{TITLE}</h1><p>{html.escape(TAGLINE)}</p></div><div class="bs-how">{how}</div></div>')
    if missing := env.missing():
        gr.Markdown(f"⚠️ **Missing environment variables:** {', '.join(missing)}. See SETUP.md.")

    with gr.Tab("App"):
        request_id = gr.State("")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                raw_input = gr.Textbox(label="Tender text", lines=15, max_lines=22,
                                       placeholder="Paste the notice: title, description, estimated value, duration, procedure, award criteria…")
                gr.HTML('<div class="bs-kicker">Try an example</div>')
                with gr.Row(elem_classes="bs-chips"):
                    example_buttons = [gr.Button(example["label"].replace("Example: ", "").capitalize(), size="sm", variant="secondary",
                                                 elem_classes="bs-chip", scale=0, min_width=0) for example in EXAMPLES]
                with gr.Accordion(settings_title(), open=False) as settings:
                    bid_cost = gr.Number(label="Cost of preparing the bid (EUR)", value=config.DEFAULT_BID_COST_EUR, minimum=0)
                    margin = gr.Slider(label="Margin on the contract value (%)", minimum=0, maximum=40, step=0.5,
                                       value=config.DEFAULT_MARGIN_PCT)
                analyse = gr.Button("Analyse", variant="primary", size="lg")
                gr.HTML('<div class="bs-foot">Emails, phone numbers, ids and contact names are removed before the text is sent '
                        "to the AI model, and the text itself is never stored. Do not paste CVs or staff lists.</div>")
            with gr.Column(scale=6):
                card = gr.HTML(PLACEHOLDER_CARD)
                with gr.Column(visible=False) as details:
                    tiles = gr.HTML()
                    found = gr.HTML()
                    with gr.Accordion("Agent steps", open=False):
                        steps = gr.Markdown("Each tool call and its result will appear here.")
                    with gr.Row():
                        up = gr.Button("👍  Useful", size="sm")
                        down = gr.Button("👎  Not useful", size="sm")
                    comment = gr.Textbox(label="Comment (optional)", lines=1, placeholder="What was wrong or missing?")
                    footer = gr.HTML(elem_classes="bs-foot")

        results = [card, details, tiles, found, steps, footer, request_id, comment]
        analyse.click(on_analyse, [raw_input, bid_cost, margin], results, api_visibility="private", show_progress="hidden")
        gr.api(run_api, api_name="run")   # the /run endpoint check.py calls: one text in, the recommendation out
        for button, example in zip(example_buttons, EXAMPLES):
            button.click(lambda text=example["text"]: text, None, raw_input, api_visibility="private", show_progress="hidden")
        for control in (bid_cost, margin):
            control.change(lambda cost, share: gr.update(label=settings_title(cost, share)), [bid_cost, margin], settings,
                           api_visibility="private", show_progress="hidden")
        up.click(lambda rid, text: on_feedback(rid, +1, text), [request_id, comment], comment, api_visibility="private")
        down.click(lambda rid, text: on_feedback(rid, -1, text), [request_id, comment], comment, api_visibility="private")

    with gr.Tab("Monitoring"):
        kpi_tiles = gr.HTML()
        alert_box = gr.HTML()
        gr.HTML('<div class="bs-section">Quality over time<span>average of the last 10 requests</span></div>')
        quality_plot = gr.LinePlot(x="time", y="value", color="signal", height=280, y_lim=[0, 1], show_label=False, visible=False)
        quality_empty = gr.HTML(_empty("The chart appears after the first request."))
        gr.HTML(f'<div class="bs-section">Drift<span>the last {monitor.WINDOW} requests against the reference set</span></div>')
        drift_table = gr.HTML(_empty("Loading…"))
        gr.HTML('<div class="bs-section">Recent requests<span>summaries only, never the tender text</span></div>')
        recent_table = gr.HTML(_empty("Loading…"))
        with gr.Accordion("Prompt bandit (bonus)" + ("" if config.BANDIT_ENABLED else ": off"), open=False):
            bandit_table = gr.Dataframe(label="Per prompt: simulated warm-up and real feedback, kept apart", interactive=False, visible=False)
            bandit_empty = gr.HTML(_empty("No prompt has been tried yet."))
        with gr.Row():
            refresh = gr.Button("Refresh", size="sm", scale=0, min_width=110)
            gr.HTML(f'<div class="bs-foot">{html.escape(LOG_TARGET)}</div>')
        monitoring = [kpi_tiles, alert_box, quality_plot, quality_empty, drift_table, recent_table, bandit_table, bandit_empty]
        refresh.click(on_refresh, None, monitoring, api_visibility="private", show_progress="hidden")
        demo.load(on_refresh, None, monitoring, api_visibility="private", show_progress="hidden")


if __name__ == "__main__":
    # Render gives the port in $PORT; locally it is 7860.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), theme=THEME, css=CSS)
