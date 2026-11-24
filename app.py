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
from bidsense import bandit, decision, features

env.load_env()
LOG_TARGET = logbook.start()

TITLE = "BidSense"
DESCRIPTION = "Paste a Spanish public tender. BidSense reads it, estimates the competition and tells you whether to bid."
EXAMPLES = json.loads((config.DATA_DIR / "examples.json").read_text(encoding="utf-8"))

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)
CSS = """
.gradio-container { max-width: 1240px !important; }
.bs-header h1 { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; }
.bs-header p { margin: 4px 0 0; opacity: .75; }
.bs-card { border-radius: 12px; padding: 20px 22px; border: 1px solid; }
.bs-card .bs-label { font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; opacity: .8; }
.bs-card .bs-decision { font-size: 2rem; font-weight: 750; letter-spacing: -0.02em; line-height: 1.15; margin: 2px 0 8px; }
.bs-card .bs-reason { font-size: .95rem; line-height: 1.45; }
.bs-card ul { margin: 6px 0 0 18px; padding: 0; }
.bs-bid { background: #ecfdf3; border-color: #a6e9c1; color: #0b5c2e; }
.bs-dont { background: #fef1f1; border-color: #f5b5b5; color: #8f1d1d; }
.bs-review { background: #fff8e6; border-color: #f2d58a; color: #7a4d00; }
.bs-neutral { background: #f3f5f8; border-color: #d6dce5; color: #354052; }
.bs-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 12px; }
.bs-tile { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 12px 14px; background: var(--background-fill-primary); }
.bs-tile .bs-k { font-size: .72rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; opacity: .65; }
.bs-tile .bs-v { font-size: 1.45rem; font-weight: 700; font-variant-numeric: tabular-nums; margin-top: 2px; }
.bs-tile .bs-s { font-size: .8rem; opacity: .7; }
.bs-alert { border-radius: 10px; padding: 12px 16px; border: 1px solid; font-size: .93rem; line-height: 1.5; }
.bs-foot { font-size: .8rem; opacity: .7; }
"""

FIELD_LABELS = {
    "cpv_code": "CPV code", "contract_type": "Contract type", "estimated_value_eur": "Estimated value",
    "duration_months": "Duration", "procedure": "Procedure", "lots_number": "Lots", "price_weight_pct": "Weight of price",
    "is_framework_agreement": "Framework agreement", "region": "Region", "buyer_type": "Buyer type", "language": "Language",
}
CARD_CLASS = {decision.BID: "bs-bid", decision.DONT_BID: "bs-dont", decision.REVIEW: "bs-review", decision.OUT_OF_SCOPE: "bs-neutral"}
EMPTY_CARD = ('<div class="bs-card bs-neutral"><div class="bs-label">Recommendation</div>'
              '<div class="bs-reason">Paste a tender and press <b>Analyse</b>.</div></div>')


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
    reasons = outcome.get("review_reasons") or []
    if outcome["decision"] == decision.REVIEW and reasons:
        body = "A person should decide:<ul>" + "".join(f"<li>{html.escape(r)}</li>" for r in reasons) + "</ul>"
    else:
        body = html.escape(outcome["reason"])
    return (f'<div class="bs-card {CARD_CLASS[outcome["decision"]]}"><div class="bs-label">Recommendation</div>'
            f'<div class="bs-decision">{decision.LABELS[outcome["decision"]]}</div><div class="bs-reason">{body}</div></div>')


def _tiles(prediction: dict | None, outcome: dict) -> str:
    if not prediction:
        return '<div class="bs-tiles">' + _tile("Expected bidders", "-") + _tile("Win chance", "-") + _tile("Expected value", "-") + "</div>"
    win, value = outcome.get("win_chance"), outcome.get("expected_value_eur")
    return '<div class="bs-tiles">' + "".join([
        _tile("Expected bidders", f"{prediction['expected_bidders']:.1f}",
              f"80% range {prediction['range_low']:.0f} to {prediction['range_high']:.0f}"),
        _tile("Win chance", f"{win:.0%}" if win is not None else "-", "1 / expected bidders"),
        _tile("Expected value", f"€{value:,.0f}" if value is not None else "-",
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


def _found(agent_output: dict) -> pd.DataFrame:
    rows = []
    for name, label in FIELD_LABELS.items():
        field = (agent_output.get("fields") or {}).get(name)
        if field is None:
            continue
        evidence = field["evidence"] + ("  (not found in the text)" if field["evidence_found"] is False else "")
        rows.append([label, _show(name, field["value"]), f"{field['confidence']:.2f}" if field["value"] is not None else "", evidence])
    return pd.DataFrame(rows, columns=["Field", "Value", "Confidence", "Evidence"])


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
    if raw_input is None or not str(raw_input).strip():
        gr.Warning("Paste the text of a tender first.")
        return [gr.update() for _ in range(7)]
    record = serve(raw_input, bid_cost_eur, margin_pct)
    outputs = record["outputs"]
    footer = (f"Latency {record['latency_s']} s · {record['tokens_in']:,} tokens in, {record['tokens_out']:,} out · "
              f"estimated cost €{record['cost_eur']:.4f} (paid-tier prices)"
              + (" · answer served from the cache" if record.get("cached") else "")
              + f" · model {record['versions']['model_version']}, prompt {record['versions']['prompt_version']}")
    return (_card(outputs["decision"]), _tiles(outputs["model"], outputs["decision"]), _found(outputs["agent"]),
            _steps(record), footer, record["id"], "")


def on_feedback(request_id, value, comment):
    if not request_id:
        gr.Warning("Analyse a tender first.")
        return gr.update()
    logbook.log_feedback(request_id, value, comment or "")
    gr.Info("Thanks for the feedback.")
    return ""


# --- Monitoring tab ----------------------------------------------------------

def _percent(value) -> str:
    return "-" if value is None else f"{value:.0%}"


def on_refresh():
    logs = logbook.load()
    reference = monitor.load_reference()
    k = monitor.kpis(logs)
    tiles = '<div class="bs-tiles">' + "".join([
        _tile("Requests", f"{k['requests']:,}"),
        _tile("Thumbs down", _percent(k["thumbs_down_share"]), f"of {k['rated']} rated"),
        _tile("Sent to review", _percent(k["review_share"])),
        _tile("Avg cost / request", "-" if k["mean_cost_eur"] is None else f"€{k['mean_cost_eur']:.4f}", "paid-tier estimate"),
        _tile("Avg latency", "-" if k["mean_latency_s"] is None else f"{k['mean_latency_s']:.1f} s"),
    ]) + "</div>"

    rule = (f"Alert when a feature's PSI is above {monitor.PSI_ALERT} (and above {monitor.NOISE_MULTIPLE}× the PSI expected by "
            f"chance for the sample size), or when more than {monitor.OTHER_LANGUAGE_ALERT:.0%} of texts are in a language "
            f"other than Spanish, Catalan, Galician or Basque. "
            f"Window: last {monitor.WINDOW} requests, at least {monitor.MIN_WINDOW}.")
    fired = monitor.alerts(logs, reference) if not logs.empty else []
    if reference is None:
        alert = ('<div class="bs-alert bs-neutral">No reference set yet: run <code>uv run python main.py reference</code> '
                 "and commit <code>monitoring/</code>.</div>")
        drift_table = pd.DataFrame(columns=monitor.DRIFT_COLUMNS)
    else:
        if fired:
            alert = ('<div class="bs-alert bs-dont"><b>Alert.</b><ul>' + "".join(f"<li>{html.escape(m)}</li>" for m in fired)
                     + f"</ul>{rule}</div>")
        elif len(logs) < monitor.MIN_WINDOW:
            alert = (f'<div class="bs-alert bs-neutral"><b>Not enough requests yet</b> ({len(logs)} of {monitor.MIN_WINDOW}) '
                     f"for the drift check. {rule}</div>")
        else:
            alert = f'<div class="bs-alert bs-bid"><b>No alerts.</b> {rule}</div>'
        current = monitor.live_features(logs) if len(logs) >= monitor.MIN_WINDOW else reference.head(0)
        drift_table = monitor.drift(reference, current)
    arms = pd.DataFrame([{"prompt": arm} | row for arm, row in bandit.state(logs).items()])
    return tiles, alert, monitor.quality(logs), drift_table, monitor.recent(logs), arms


with gr.Blocks(title=TITLE) as demo:
    gr.HTML(f'<div class="bs-header"><h1>{TITLE}</h1><p>{DESCRIPTION}</p></div>')
    if missing := env.missing():
        gr.Markdown(f"⚠️ **Missing environment variables:** {', '.join(missing)}. See SETUP.md.")

    with gr.Tab("App"):
        request_id = gr.State("")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                raw_input = gr.Textbox(label="Paste the tender text", lines=16, max_lines=24,
                                       placeholder="Título, descripción, valor estimado, duración, procedimiento, criterios de adjudicación…")
                with gr.Row():
                    example_buttons = [gr.Button(example["label"], size="sm", variant="secondary") for example in EXAMPLES]
                with gr.Accordion("Company settings", open=False):
                    bid_cost = gr.Number(label="Cost of preparing the bid (EUR)", value=config.DEFAULT_BID_COST_EUR, minimum=0)
                    margin = gr.Slider(label="Margin on the contract value (%)", minimum=0, maximum=40, step=0.5,
                                       value=config.DEFAULT_MARGIN_PCT)
                analyse = gr.Button("Analyse", variant="primary", size="lg")
                gr.HTML('<div class="bs-foot">Emails, phone numbers, ids and contact names are removed before the text is sent '
                        "to the AI model, and the text itself is never stored. Do not paste CVs or staff lists.</div>")
            with gr.Column(scale=6):
                card = gr.HTML(EMPTY_CARD)
                tiles = gr.HTML()
                found = gr.Dataframe(label="What the agent found", headers=["Field", "Value", "Confidence", "Evidence"],
                                     interactive=False, wrap=True, column_widths=["21%", "19%", "14%", "46%"])
                with gr.Accordion("Agent steps", open=False):
                    steps = gr.Markdown("Each tool call and its result will appear here.")
                with gr.Row():
                    up = gr.Button("👍  Useful", size="sm")
                    down = gr.Button("👎  Not useful", size="sm")
                comment = gr.Textbox(label="Comment (optional)", lines=1, placeholder="What was wrong or missing?")
                footer = gr.HTML(elem_classes="bs-foot")

        results = [card, tiles, found, steps, footer, request_id, comment]
        analyse.click(on_analyse, [raw_input, bid_cost, margin], results, api_visibility="private")
        gr.api(run_api, api_name="run")   # the /run endpoint check.py calls: one text in, the recommendation out
        for button, example in zip(example_buttons, EXAMPLES):
            button.click(lambda text=example["text"]: text, None, raw_input, api_visibility="private")
        up.click(lambda rid, text: on_feedback(rid, +1, text), [request_id, comment], comment, api_visibility="private")
        down.click(lambda rid, text: on_feedback(rid, -1, text), [request_id, comment], comment, api_visibility="private")

    with gr.Tab("Monitoring"):
        refresh = gr.Button("Refresh", size="sm")
        kpi_tiles = gr.HTML()
        alert_box = gr.HTML()
        quality_plot = gr.LinePlot(x="time", y="value", color="signal", height=300, y_lim=[0, 1],
                                   label="Quality over time (average of the last 10 requests)")
        drift_table = gr.Dataframe(
            label=f"Drift: last {monitor.WINDOW} requests against monitoring/reference_features.csv", interactive=False)
        recent_table = gr.Dataframe(label="Recent requests (summaries only, never the tender text)", interactive=False, wrap=True)
        with gr.Accordion("Prompt bandit (bonus)" + ("" if config.BANDIT_ENABLED else ": off"), open=False):
            bandit_table = gr.Dataframe(label="Per prompt: simulated warm-up and real feedback, kept apart", interactive=False)
        gr.HTML(f'<div class="bs-foot">{html.escape(LOG_TARGET)}</div>')
        monitoring = [kpi_tiles, alert_box, quality_plot, drift_table, recent_table, bandit_table]
        refresh.click(on_refresh, None, monitoring, api_visibility="private")
        demo.load(on_refresh, None, monitoring, api_visibility="private")


if __name__ == "__main__":
    # Render gives the port in $PORT; locally it is 7860.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), theme=THEME, css=CSS)
