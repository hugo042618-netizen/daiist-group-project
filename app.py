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
import urllib.parse

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
TAGLINE = "Paste a Spanish public tender. Get a bid decision in seconds."
EXAMPLES = json.loads((config.DATA_DIR / "examples.json").read_text(encoding="utf-8"))

# --- Look: "the tender dossier" ------------------------------------------------
# A clean official form on paper; the decision lands on it as a rubber stamp. Light theme only (it is shown
# on a projector). Nothing here is fetched from outside: the two fonts are in assets/fonts/ (SIL Open Font
# License) and the app serves them itself.
PAPER, INK, LINE, MUTED = "#F3F5F2", "#1E2B3A", "#C9D1D9", "#4B5868"
STAMP_INK = {decision.BID: "#1F7A4D", decision.DONT_BID: "#B3261E", decision.REVIEW: "#5A3E9B", decision.OUT_OF_SCOPE: "#6E7781"}
STAMP_CLASS = {decision.BID: "bs-ink-bid", decision.DONT_BID: "bs-ink-dont", decision.REVIEW: "bs-ink-review", decision.OUT_OF_SCOPE: "bs-ink-out"}
STAMP_WORD = {decision.BID: "BID", decision.DONT_BID: "DON'T BID", decision.REVIEW: "REVIEW", decision.OUT_OF_SCOPE: "OUT OF SCOPE"}

FONT_DIR = config.ROOT / "assets" / "fonts"
gr.set_static_paths(paths=[FONT_DIR])
_font = lambda name: f"/gradio_api/file={(FONT_DIR / name).as_posix()}"  # noqa: E731

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.slate, radius_size=gr.themes.sizes.radius_sm,
    font=["Public Sans", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
    font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
).set(
    body_background_fill=PAPER, body_text_color=INK, body_text_color_subdued=MUTED,
    background_fill_primary="#FFFFFF", background_fill_secondary=PAPER,
    border_color_primary=LINE, block_border_color=LINE, block_background_fill="transparent",
    block_label_text_color=MUTED, block_title_text_color=INK, block_label_background_fill="transparent",
    block_shadow="none", shadow_drop="none", shadow_drop_lg="none",
    input_background_fill="#FFFFFF", input_border_color=LINE, input_border_color_focus=INK, input_shadow="none", input_shadow_focus="none",
    button_primary_background_fill=INK, button_primary_background_fill_hover="#2C3D52", button_primary_text_color="#FFFFFF",
    button_primary_border_color=INK,
    button_secondary_background_fill="#FFFFFF", button_secondary_background_fill_hover=PAPER, button_secondary_text_color=INK,
    button_secondary_border_color=LINE,
    color_accent=INK, color_accent_soft="#E4E9EE", slider_color=INK, link_text_color=INK,
)
# The app is light only: a visitor whose system is dark is sent to the same page with Gradio's light theme.
HEAD = """<meta name="color-scheme" content="light">
<script>(function(){var u=new URL(window.location.href);if(u.searchParams.get("__theme")!=="light"){u.searchParams.set("__theme","light");window.location.replace(u.href);}})();</script>"""
# The stamp's ink texture: a noise mask, so the ink is slightly uneven, as on paper.
_NOISE = ("data:image/svg+xml," + urllib.parse.quote(
    "<svg xmlns='http://www.w3.org/2000/svg' width='260' height='260'><filter id='n' x='0' y='0' width='100%' height='100%'>"
    "<feTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2' seed='11'/>"
    "<feColorMatrix type='matrix' values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 -1.9 1.92'/></filter>"
    "<rect width='260' height='260' filter='url(#n)'/></svg>"))
CSS = """
@font-face { font-family: "Public Sans"; src: url("__PUBLIC_SANS__") format("woff2"); font-weight: 100 900; font-style: normal; font-display: swap; }
@font-face { font-family: "Big Shoulders Display"; src: url("__BIG_SHOULDERS__") format("woff2"); font-weight: 100 900; font-style: normal; font-display: swap; }
:root { color-scheme: light; --bs-paper: __PAPER__; --bs-ink: __INK__; --bs-line: __LINE__; --bs-muted: __MUTED__;
  --bs-bid: #1F7A4D; --bs-dont: #B3261E; --bs-review: #5A3E9B; --bs-out: #6E7781; --bs-ease: cubic-bezier(0.23, 1, 0.32, 1); }
body, .gradio-container { background: var(--bs-paper) !important; color: var(--bs-ink); }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; font-family: "Public Sans", system-ui, sans-serif !important; }
footer { display: none !important; }
.gradio-container *:focus-visible { outline: 2px solid var(--bs-ink) !important; outline-offset: 2px !important; }
.gradio-container button { transition: transform 140ms var(--bs-ease), background-color 140ms ease, border-color 140ms ease; }
.gradio-container button:active { transform: scale(0.97); }

.bs-head { padding: 10px 2px 4px; border-bottom: 2px solid var(--bs-ink); margin-bottom: 4px; }
.bs-wordmark { display: flex; align-items: center; gap: 11px; font-family: "Big Shoulders Display", "Public Sans", sans-serif; font-weight: 800;
  font-size: 2.35rem; line-height: 1; letter-spacing: .01em; color: var(--bs-ink); }
.bs-mark { width: 25px; height: 25px; border: 3px solid var(--bs-ink); outline: 1.5px solid var(--bs-ink); outline-offset: -7px;
  border-radius: 3px; transform: rotate(-4deg); flex: 0 0 auto; }
.bs-head p { margin: 7px 0 10px; font-size: 1.02rem; color: var(--bs-muted); }

.bs-sheet { background: #fff !important; border: 1px solid var(--bs-line) !important; border-radius: 2px !important; padding: 18px 22px 14px !important;
  box-shadow: 3px 3px 0 #E2E6E3 !important; }
.bs-sheet textarea { border: 0 !important; box-shadow: none !important; background: transparent !important; padding: 6px 0 !important;
  line-height: 1.55 !important; font-size: .95rem !important; color: var(--bs-ink) !important; }
.bs-sheet label > span, .bs-sheet [data-testid="block-info"] { font-weight: 650 !important; color: var(--bs-ink) !important; font-size: .9rem !important; }
.bs-chips { gap: 8px !important; flex-wrap: wrap !important; align-items: center !important; justify-content: flex-start !important; }
.bs-chips > * { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
.bs-chips .bs-try { font-size: .88rem; color: var(--bs-muted); margin-right: 2px; }
button.bs-chip { border-radius: 2px !important; padding: 5px 12px !important; font-weight: 550 !important; font-size: .85rem !important;
  border: 1px solid var(--bs-ink) !important; background: transparent !important; color: var(--bs-ink) !important; min-height: 0 !important; }
button.bs-chip:hover { background: #fff !important; }
button.bs-go { border-radius: 2px !important; font-weight: 650 !important; font-size: 1.02rem !important; letter-spacing: 0 !important; }
.bs-foot { font-size: .8rem; color: var(--bs-muted); line-height: 1.45; }
.gradio-container button.secondary { border: 1px solid var(--bs-ink) !important; background: #fff !important; color: var(--bs-ink) !important; border-radius: 2px !important; }
.gradio-container button.secondary:hover { background: var(--bs-paper) !important; }
.gradio-container .form, .gradio-container .block.bs-plain { background: transparent !important; border: 0 !important; box-shadow: none !important; }
.gradio-container .block.bs-plain { padding: 0 !important; }
.bs-plain label > span, .bs-plain [data-testid="block-info"] { font-size: .86rem !important; color: var(--bs-ink) !important; }
.bs-plain textarea, .bs-plain input { border: 1px solid #9AA5B1 !important; border-radius: 2px !important; background: #fff !important; }

.bs-stamp-area { min-height: 172px; display: flex; align-items: center; justify-content: center; padding: 18px 10px 14px; }
.bs-stamp { display: inline-block; min-width: min(320px, 100%); max-width: 100%; padding: 15px 34px 13px; text-align: center; border: 4px solid currentColor;
  outline: 1.5px solid currentColor; outline-offset: -10px; border-radius: 7px; transform: rotate(-4deg); mix-blend-mode: multiply;
  -webkit-mask-image: url("__NOISE__"); mask-image: url("__NOISE__"); -webkit-mask-size: 260px 260px; mask-size: 260px 260px;
  animation: bs-press 250ms var(--bs-ease) both; }
.bs-stamp-word { font-family: "Big Shoulders Display", "Public Sans", sans-serif; font-weight: 900; font-size: clamp(2.7rem, 6.2vw, 4.3rem);
  line-height: .98; letter-spacing: .03em; text-transform: uppercase; color: inherit; white-space: nowrap; }
.bs-stamp-line { margin-top: 8px; padding-top: 6px; border-top: 1.5px solid currentColor; font-weight: 700; font-size: 1.2rem; line-height: 1.25;
  font-variant-numeric: tabular-nums; color: inherit; }
.bs-ink-bid { color: var(--bs-bid); } .bs-ink-dont { color: var(--bs-dont); } .bs-ink-review { color: var(--bs-review); } .bs-ink-out { color: var(--bs-out); }
@keyframes bs-press { from { transform: rotate(-4deg) scale(1.12); opacity: .25; } to { transform: rotate(-4deg) scale(1); opacity: 1; } }
.bs-stamp-empty { border: 2px dashed #8C97A3; outline: 0; color: var(--bs-muted); mix-blend-mode: normal; -webkit-mask-image: none; mask-image: none;
  animation: none; padding: 34px 30px; }
.bs-stamp-empty .bs-stamp-hint { font-size: 1.05rem; font-weight: 600; color: var(--bs-muted); }
.bs-stamp-empty small { display: block; margin-top: 4px; font-size: .86rem; font-weight: 400; color: var(--bs-muted); }

.bs-progress { border-left: 3px solid var(--bs-ink); padding: 6px 0 6px 18px; margin: 34px 6px; }
.bs-progress ol { list-style: none; margin: 0; padding: 0; display: grid; gap: 9px; }
.bs-progress li { font-size: 1.1rem; font-weight: 600; color: var(--bs-ink); opacity: .3; animation: bs-step 350ms ease-out forwards; }
.bs-progress li:nth-child(1) { animation-delay: 0ms; } .bs-progress li:nth-child(2) { animation-delay: 1800ms; } .bs-progress li:nth-child(3) { animation-delay: 3200ms; }
@keyframes bs-step { to { opacity: 1; } }

.bs-ledger { display: grid; margin: 0; border-top: 2px solid var(--bs-ink); border-bottom: 1px solid var(--bs-ink); background: #fff; }
.bs-ledger.bs-three { grid-template-columns: repeat(3, minmax(0, 1fr)); } .bs-ledger.bs-five { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.bs-ledger > div { padding: 12px 16px 11px; border-left: 1px solid var(--bs-line); min-width: 0; }
.bs-ledger > div:first-child { border-left: 0; }
.bs-ledger dd { margin: 0; white-space: nowrap; font-size: 1.95rem; font-weight: 700; letter-spacing: -.01em; line-height: 1.1; font-variant-numeric: tabular-nums; color: var(--bs-ink); }
.bs-ledger dt { margin-top: 4px; font-size: .86rem; font-weight: 600; color: var(--bs-ink); }
.bs-ledger span { display: block; font-size: .8rem; color: var(--bs-muted); }
.bs-reason { margin: 4px 2px 0; font-size: 1rem; line-height: 1.5; color: var(--bs-ink); max-width: 68ch; }
.bs-reason ul { margin: 4px 0 0 20px; padding: 0; }

.bs-h { font-size: 1.02rem; font-weight: 700; margin: 6px 2px 2px; color: var(--bs-ink); }
.bs-legend { margin: 0 2px 9px; font-size: .84rem; color: var(--bs-muted); }
.bs-form { display: grid; grid-template-columns: repeat(auto-fill, minmax(196px, 1fr)); gap: 9px; }
.bs-field { position: relative; background: #fff; border: 1px solid #9AA5B1; border-radius: 2px; padding: 7px 10px 8px; min-height: 58px; cursor: help; }
.bs-field.bs-low { border: 1.5px dashed var(--bs-ink); }
.bs-field:focus { outline: 2px solid var(--bs-ink); outline-offset: 2px; }
.gradio-container textarea::placeholder, .gradio-container input::placeholder { color: var(--bs-muted) !important; opacity: 1 !important; }
.bs-flabel { display: block; font-size: .78rem; color: var(--bs-muted); padding-right: 40px; }
.bs-fconf { position: absolute; top: 7px; right: 10px; font-size: .78rem; font-variant-numeric: tabular-nums; color: var(--bs-muted); }
.bs-fvalue { display: block; margin-top: 2px; font-size: 1rem; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--bs-ink); overflow-wrap: anywhere; }
.bs-fvalue small { display: block; font-size: .78rem; font-weight: 400; color: var(--bs-muted); }
.bs-fvalue.bs-none { font-weight: 400; color: var(--bs-muted); }
.bs-miss { display: inline-block; margin-top: 4px; border: 1px solid var(--bs-ink); border-radius: 2px; padding: 0 5px; font-size: .72rem; font-weight: 600; color: var(--bs-ink); }
.bs-quote { display: none; position: absolute; z-index: 30; left: -1px; top: calc(100% + 5px); width: max-content; max-width: min(340px, 86vw);
  background: var(--bs-ink); color: #fff; padding: 8px 11px; border-radius: 2px; font-size: .84rem; line-height: 1.4; font-weight: 400; }
.bs-field:hover .bs-quote, .bs-field:focus .bs-quote, .bs-field:focus-within .bs-quote { display: block; }
.gradio-container .block:has(.bs-form), .gradio-container .html-container:has(.bs-form), .gradio-container .prose:has(.bs-form) { overflow: visible !important; }

.bs-notice { background: #fff; border: 1px solid var(--bs-line); border-left: 6px solid #6E7781; padding: 12px 16px 12px 16px; }
.bs-notice.bs-ok { border-left-color: var(--bs-bid); } .bs-notice.bs-alert { border-left-color: var(--bs-dont); } .bs-notice.bs-wait { border-left-color: #6E7781; }
.bs-notice b { display: block; font-size: 1.05rem; font-weight: 700; color: var(--bs-ink); }
.bs-notice div, .bs-notice li { font-size: .93rem; line-height: 1.5; color: var(--bs-ink); }
.bs-notice ul { margin: 4px 0 4px 18px; padding: 0; }
.bs-notice small { display: block; margin-top: 5px; font-size: .8rem; color: var(--bs-muted); }
table.bs-table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: .9rem; margin: 0; background: #fff; border: 0 !important;
  border-top: 2px solid var(--bs-ink) !important; border-bottom: 1px solid var(--bs-ink) !important; }
table.bs-table thead, table.bs-table tbody, table.bs-table tr { border: 0 !important; background: transparent !important; }
table.bs-table th { text-align: left; font-size: .8rem; font-weight: 650; padding: 8px 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  color: var(--bs-ink); background: transparent; border: 0 !important; border-bottom: 1px solid var(--bs-ink) !important; }
table.bs-table td { padding: 8px 12px; vertical-align: middle; color: var(--bs-ink); overflow-wrap: anywhere; border: 0 !important;
  border-bottom: 1px solid var(--bs-line) !important; }
table.bs-table tr:last-child td { border-bottom: 0 !important; }
table.bs-table td.bs-f { font-weight: 600; } table.bs-table td.bs-num { font-variant-numeric: tabular-nums; white-space: nowrap; overflow-wrap: normal; }
table.bs-table td.bs-quiet { color: var(--bs-muted); font-size: .86rem; font-variant-numeric: tabular-nums; }
.bs-pill { display: inline-block; background: #fff; border: 1.5px solid currentColor; border-radius: 3px; padding: 0 7px 1px; white-space: nowrap;
  font-family: "Big Shoulders Display", "Public Sans", sans-serif; font-weight: 800; font-size: .95rem; letter-spacing: .04em; text-transform: uppercase; line-height: 1.35; }
.bs-tag { display: inline-block; background: #fff; border: 1px solid currentColor; border-radius: 2px; padding: 0 6px; font-size: .78rem; font-weight: 650; }
.bs-tag.bs-ok { color: var(--bs-bid); } .bs-tag.bs-alert { color: var(--bs-dont); }
.bs-empty { border: 1.5px dashed #8C97A3; padding: 22px 18px; text-align: center; color: var(--bs-ink); font-weight: 600; background: transparent; }
.bs-empty span { display: block; margin-top: 3px; font-weight: 400; font-size: .86rem; color: var(--bs-muted); }
@media (max-width: 760px) {
  .bs-ledger.bs-five { grid-template-columns: repeat(2, minmax(0, 1fr)); } .bs-ledger.bs-five > div:nth-child(odd) { border-left: 0; }
  .bs-ledger.bs-five > div { border-top: 1px solid var(--bs-line); } .bs-ledger dd { font-size: 1.3rem; }
  .bs-ledger > div { padding: 10px 10px 9px; } .bs-wordmark { font-size: 2rem; } .bs-stamp { padding: 13px 22px 11px; }
  table.bs-table { font-size: .82rem; } table.bs-table th, table.bs-table td { padding: 7px 6px; }
}
@media (prefers-reduced-motion: reduce) {
  .bs-stamp, .bs-progress li { animation: none; } .bs-progress li { opacity: 1; } .gradio-container button:active { transform: none; }
}
""".replace("__PUBLIC_SANS__", _font("PublicSans-latin.woff2")).replace("__BIG_SHOULDERS__", _font("BigShouldersDisplay-latin.woff2")) \
   .replace("__PAPER__", PAPER).replace("__INK__", INK).replace("__LINE__", LINE).replace("__MUTED__", MUTED).replace("__NOISE__", _NOISE)

FIELD_LABELS = {
    "cpv_code": "CPV code", "contract_type": "Contract type", "estimated_value_eur": "Estimated value",
    "duration_months": "Duration", "procedure": "Procedure", "lots_number": "Lots", "price_weight_pct": "Weight of price",
    "is_framework_agreement": "Framework agreement", "region": "Region", "buyer_type": "Buyer type", "language": "Language",
}
PLACEHOLDER_CARD = ('<div class="bs-stamp-area"><div class="bs-stamp bs-stamp-empty"><div class="bs-stamp-hint">Your decision will appear here'
                    "<small>Paste a tender, or try an example, and press Analyse tender.</small></div></div></div>")
# The three lines light up on a timer while the request runs; they are a progress hint, not a measurement.
LOADING_CARD = ('<div class="bs-progress" role="status"><ol><li>Reading the tender…</li><li>Estimating competition…</li>'
                "<li>Deciding…</li></ol></div>")


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

def _euros(value: float) -> str:
    """An amount for the stamp and the ledger, with a real minus sign in front of the euro sign."""
    return f"−€{abs(value):,.0f}" if value < 0 else f"€{value:,.0f}"


def _figure(value: str, label: str, note: str = "") -> str:
    """One figure of a ledger row: the number, its label underneath, and a small note."""
    return (f"<div><dd>{html.escape(value)}</dd><dt>{html.escape(label)}</dt>"
            + (f"<span>{html.escape(note)}</span>" if note else "") + "</div>")


def _card(outcome: dict) -> str:
    """The verdict as a rubber stamp: the word, and the expected value on a small line inside."""
    verdict, value = outcome["decision"], outcome.get("expected_value_eur")
    if verdict == decision.OUT_OF_SCOPE:
        line = "Not a tender"
    elif verdict == decision.REVIEW:
        line = "A person decides"
    else:
        line = f"Expected value {_euros(value)}" if value is not None else ""
    return (f'<div class="bs-stamp-area"><div class="bs-stamp {STAMP_CLASS[verdict]}" role="status" aria-label="Decision: {html.escape(STAMP_WORD[verdict], quote=True)}">'
            f'<div class="bs-stamp-word">{html.escape(STAMP_WORD[verdict])}</div>'
            + (f'<div class="bs-stamp-line">{html.escape(line)}</div>' if line else "") + "</div></div>")


def _tiles(prediction: dict | None, outcome: dict) -> str:
    """The three figures as a ledger row."""
    if not prediction:
        return ('<dl class="bs-ledger bs-three">' + _figure("–", "Expected bidders", "no prediction") + _figure("–", "Win chance")
                + _figure("–", "Expected value") + "</dl>")
    win, value = outcome.get("win_chance"), outcome.get("expected_value_eur")
    return '<dl class="bs-ledger bs-three">' + "".join([
        _figure(f"{prediction['expected_bidders']:.1f}", "Expected bidders",
                f"80% range {prediction['range_low']:.0f} to {prediction['range_high']:.0f}"),
        _figure(f"{win:.0%}" if win is not None else "–", "Win chance", "1 / expected bidders"),
        _figure(_euros(value) if value is not None else "–", "Expected value",
                f"margin {outcome.get('margin_pct', 0):g}%, bid cost €{outcome.get('bid_cost_eur', 0):,.0f}" if value is not None else "value missing"),
    ]) + "</dl>"


def _reason(outcome: dict) -> str:
    """The reason sentence, as the pipeline wrote it."""
    reasons = outcome.get("review_reasons") or []
    if outcome["decision"] == decision.REVIEW and reasons:
        return '<div class="bs-reason">A person should decide:<ul>' + "".join(f"<li>{html.escape(r)}</li>" for r in reasons) + "</ul></div>"
    return f'<p class="bs-reason">{html.escape(outcome["reason"])}</p>'


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



def _found(agent_output: dict, text: str | None = None) -> str:
    """"What the agent found" as a filled-in form: one boxed field per fact, with its confidence. The quote the
    fact came from shows on hover and on keyboard focus. A dashed box is a field the app does not trust
    (confidence below config.MIN_CONFIDENCE). With `text` (what the agent read), a quote that is not in it gets
    a "not found in text" mark: for the reader only, it changes no confidence and no decision."""
    boxes = []
    for name, label in FIELD_LABELS.items():
        field = (agent_output.get("fields") or {}).get(name)
        if field is None:
            continue
        has_value = field["value"] is not None
        confidence = float(field.get("confidence") or 0)
        value = html.escape(_show(name, field["value"]))
        if name == "cpv_code" and field["value"] in tools.cpv_codes():
            value += f'<small>{html.escape(tools.cpv_codes()[field["value"]]["en"])}</small>'
        quote = str(field.get("evidence") or "").strip()
        missing = evidence.in_text(quote, text) is False if text is not None else field.get("evidence_found") is False
        mark = '<span class="bs-miss">not found in text</span>' if quote and missing else ""
        if quote:
            tip = f"“{html.escape(quote)}”" + (" This quote does not appear in the text." if mark else "")
        else:
            tip = "Inferred: the agent gave no quote for it." if has_value else "The text does not say."
        low = has_value and confidence < config.MIN_CONFIDENCE
        boxes.append(
            f'<div class="bs-field{" bs-low" if low else ""}" tabindex="0">'
            f'<span class="bs-flabel">{html.escape(label)}</span>'
            + (f'<span class="bs-fconf" aria-label="confidence {confidence:.0%}">{confidence:.0%}</span>' if has_value else "")
            + f'<span class="bs-fvalue{"" if has_value else " bs-none"}">{value}</span>{mark}'
            f'<span class="bs-quote" role="tooltip">{tip}</span></div>')
    if not boxes:
        return '<div class="bs-h">What the agent found</div><div class="bs-empty">Nothing was extracted<span>The agent did not read any field from this text.</span></div>'
    return ('<div class="bs-h">What the agent found</div>'
            f'<p class="bs-legend">Hover or focus a field to read the quote it came from. The percentage is the agent’s confidence; '
            f"a dashed box is below {config.MIN_CONFIDENCE:.0%}.</p>"
            '<div class="bs-form">' + "".join(boxes) + "</div>")


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
    """First the progress lines, then the result. Outputs: card, details column, figures, reason, facts, steps, footer,
    request id, comment."""
    if raw_input is None or not str(raw_input).strip():
        gr.Warning("Paste the text of a tender first.")
        yield [gr.update() for _ in range(9)]
        return
    yield [LOADING_CARD, gr.update(visible=False)] + [gr.update() for _ in range(7)]
    record = serve(raw_input, bid_cost_eur, margin_pct)
    outputs = record["outputs"]
    footer = (f"Latency {record['latency_s']} s · {record['tokens_in']:,} tokens in, {record['tokens_out']:,} out · "
              f"estimated cost €{record['cost_eur']:.4f} (paid-tier prices)"
              + (" · answer served from the cache" if record.get("cached") else "")
              + f" · model {record['versions']['model_version']}, prompt {record['versions']['prompt_version']}")
    # An out-of-scope text has no prediction and no extracted facts worth a confidence: only the stamp and the reason.
    in_scope = record["decision"] != decision.OUT_OF_SCOPE
    read = redact(pipeline.read_input(raw_input))[0][: config.MAX_INPUT_CHARS]   # what the agent read, for the evidence marks
    yield [_card(outputs["decision"]), gr.update(visible=True),
           gr.update(value=_tiles(outputs["model"], outputs["decision"]) if in_scope else "", visible=in_scope),
           _reason(outputs["decision"]),
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
THUMBS_DOWN_SIGNAL = "share thumbs down (of rated)"
SIGNAL_COLOURS = {"share sent to review": STAMP_INK[decision.REVIEW], "mean confidence": INK, THUMBS_DOWN_SIGNAL: STAMP_INK[decision.DONT_BID]}


def _percent(value) -> str:
    return "–" if value is None else f"{value:.0%}"


def _plain(text: str) -> str:
    """Monitoring texts without emoji (the only emoji of the interface are on the two feedback buttons)."""
    return str(text).replace("share 👎 (of rated)", THUMBS_DOWN_SIGNAL).replace("👎", "thumbs down").replace("👍", "thumbs up")


def _empty(why: str) -> str:
    return f'<div class="bs-empty">No data yet<span>{html.escape(why)}</span></div>'


def _banner(kind: str, title: str, body: str, rule: str = "") -> str:
    """A notice strip with a coloured left rule: bs-wait (grey), bs-ok (green) or bs-alert (red)."""
    return (f'<div class="bs-notice {kind}" role="status"><b>{html.escape(title)}</b>{body}'
            + (f"<small>{html.escape(rule)}</small>" if rule else "") + "</div>")


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
    return '<dl class="bs-ledger bs-five">' + "".join([
        _figure(f"{k['requests']:,}", "Requests", "since the first request"),
        _figure(_percent(k["review_share"]), "Sent to review", "of all requests"),
        _figure(_percent(k["thumbs_down_share"]), "Thumbs down", f"of {k['rated']} rated"),
        _figure("–" if k["mean_cost_eur"] is None else f"€{k['mean_cost_eur']:.4f}", "Average cost", "per request, paid-tier estimate"),
        _figure("–" if k["mean_latency_s"] is None else f"{k['mean_latency_s']:.1f} s", "Average latency", "per request"),
    ]) + "</dl>"


def _drift_banner(logs: pd.DataFrame, reference) -> str:
    rule = (f"Alert when a feature's PSI is above {monitor.PSI_ALERT} and above {monitor.NOISE_MULTIPLE}× the PSI expected by chance, "
            f"when more than {monitor.OTHER_LANGUAGE_ALERT:.0%} of texts are not in Spanish, Catalan, Galician or Basque, or when more "
            f"than {monitor.THUMBS_DOWN_ALERT:.0%} of rated requests are thumbs down. Window: the last {monitor.WINDOW} requests.")
    if reference is None:
        return _banner("bs-wait", "No reference set yet", "<div>Run <code>uv run python main.py reference</code> and commit "
                       "<code>monitoring/</code>.</div>")
    if len(logs) < monitor.MIN_WINDOW:
        return _banner("bs-wait", "Drift check: waiting for data",
                       f"<div>{len(logs)} of the {monitor.MIN_WINDOW} requests it needs before it can say anything.</div>", rule)
    fired = monitor.alerts(logs, reference)
    if fired:
        return _banner("bs-alert", "Alert", "<ul>" + "".join(f"<li>{html.escape(_plain(m))}</li>" for m in fired) + "</ul>", rule)
    return _banner("bs-ok", "No drift", f"<div>The last {min(len(logs), monitor.WINDOW)} requests look like the reference set, "
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
        badge = '<span class="bs-tag bs-alert">alert</span>' if row.alert else '<span class="bs-tag bs-ok">ok</span>'
        rows.append(f'<tr><td class="bs-f">{html.escape(DRIFT_LABELS.get(row.feature, row.feature))}</td><td>{html.escape(row.metric)}</td>'
                    f'<td class="bs-num">{value}</td><td class="bs-num">{limit}</td><td>{badge}</td></tr>')
    return ('<table class="bs-table"><thead><tr><th>Feature</th><th>Measure</th><th>Value</th>'
            "<th>Alert above</th><th>Status</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _recent_table(logs: pd.DataFrame) -> str:
    """The latest requests as a ruled ledger: summaries only, never the tender text."""
    recent = monitor.recent(logs)
    if recent.empty:
        return _empty("Analysed tenders are listed here (summaries only, never the text).")
    rows = []
    for row in recent.to_dict("records"):
        word = str(row["decision"] or "")
        pill = (f'<span class="bs-pill {STAMP_CLASS[word]}">{html.escape(STAMP_WORD[word])}</span>' if word in STAMP_WORD
                else html.escape(word) or "–")
        bidders = row["expected bidders"]
        confidence = pd.to_numeric(row["confidence"], errors="coerce")   # monitor.recent() leaves it empty for out-of-scope texts
        latency = pd.to_numeric(row["latency (s)"], errors="coerce")
        rated = {"👍": "Up", "👎": "Down"}.get(str(row["feedback"]), "–")
        rows.append(
            f'<tr><td class="bs-quiet">{html.escape(str(row["time (UTC)"]))}</td><td>{html.escape(str(row["input summary"]))}</td>'
            f"<td>{pill}</td>"
            f'<td class="bs-num">{"–" if bidders in ("", None) or pd.isna(bidders) else f"{float(bidders):.1f}"}</td>'
            f'<td class="bs-num">{"–" if pd.isna(confidence) else f"{confidence:.0%}"}</td>'
            f'<td class="bs-num">{"–" if pd.isna(latency) else f"{latency:.1f} s"}</td>'
            f'<td class="bs-num">{"–" if row["cost (€)"] == "" else f"€{float(row['cost (€)']):.4f}"}</td>'
            f"<td>{rated}</td></tr>")
    return ('<table class="bs-table"><colgroup><col style="width:12%"><col style="width:28%"><col style="width:15%">'
            '<col style="width:8%"><col style="width:11%"><col style="width:9%"><col style="width:10%"><col style="width:7%"></colgroup>'
            "<thead><tr><th>Time (UTC)</th><th>Input summary</th><th>Decision</th><th>Bidders</th><th>Confidence</th><th>Latency</th>"
            "<th>Cost</th><th>Rated</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _quality(logs: pd.DataFrame) -> pd.DataFrame:
    quality = monitor.quality(logs)
    return quality.assign(signal=quality["signal"].map(_plain)) if not quality.empty else quality


def on_refresh():
    """Every panel of the Monitoring tab. Each one is built on its own, so one failure cannot blank the page."""
    logs = _safely("logs", logbook.load, pd.DataFrame())
    reference = _safely("reference", monitor.load_reference, None)
    quality = _safely("quality", lambda: _quality(logs), pd.DataFrame(columns=["time", "signal", "value"]))
    arms = _safely("bandit", lambda: pd.DataFrame([{"prompt": arm} | row for arm, row in bandit.state(logs).items()]), pd.DataFrame())
    return (
        _safely("kpis", lambda: _kpi_tiles(logs), _kpi_tiles(pd.DataFrame())),
        _safely("banner", lambda: _drift_banner(logs, reference), _banner("bs-wait", "No data yet", "<div>The drift check could not run.</div>")),
        gr.update(value=quality, visible=not quality.empty),
        gr.update(value=_empty("The chart appears after the first request."), visible=quality.empty),
        _safely("drift table", lambda: _drift_table(logs, reference), _empty("The drift table could not be computed.")),
        _safely("recent", lambda: _recent_table(logs), _empty("The recent requests could not be listed.")),
        gr.update(value=arms, visible=not arms.empty),
        gr.update(value=_empty("No prompt has been tried yet."), visible=arms.empty),
    )


with gr.Blocks(title=TITLE) as demo:
    gr.HTML(f'<header class="bs-head"><div class="bs-wordmark"><span class="bs-mark" aria-hidden="true"></span>{TITLE}</div>'
            f"<p>{html.escape(TAGLINE)}</p></header>")
    if missing := env.missing():
        gr.Markdown(f"**Missing environment variables:** {', '.join(missing)}. See SETUP.md.")

    with gr.Tab("App"):
        request_id = gr.State("")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320):
                raw_input = gr.Textbox(label="Tender text", lines=11, max_lines=18, elem_classes="bs-sheet",
                                       placeholder="Paste the notice: title, description, estimated value, duration, procedure, award criteria…")
                with gr.Row(elem_classes="bs-chips"):
                    gr.HTML('<span class="bs-try">Try an example</span>', scale=0, min_width=0)
                    example_buttons = [gr.Button(example["label"].replace("Example: ", "").capitalize(), size="sm", variant="secondary",
                                                 elem_classes="bs-chip", scale=0, min_width=0) for example in EXAMPLES]
                analyse = gr.Button("Analyse tender", variant="primary", size="lg", elem_classes="bs-go")
                with gr.Accordion(settings_title(), open=False) as settings:
                    bid_cost = gr.Number(label="Cost of preparing the bid (EUR)", value=config.DEFAULT_BID_COST_EUR, minimum=0)
                    margin = gr.Slider(label="Margin on the contract value (%)", minimum=0, maximum=40, step=0.5,
                                       value=config.DEFAULT_MARGIN_PCT)
                gr.HTML('<div class="bs-foot">Emails, phone numbers, ids and contact names are removed before the text is sent '
                        "to the AI model, and the text itself is never stored. Do not paste CVs or staff lists.</div>")
            with gr.Column(scale=6, min_width=320):
                card = gr.HTML(PLACEHOLDER_CARD)
                with gr.Column(visible=False) as details:
                    tiles = gr.HTML()
                    reason = gr.HTML()
                    found = gr.HTML()
                    with gr.Accordion("Agent steps", open=False):
                        steps = gr.Markdown("Each tool call and its result will appear here.")
                    with gr.Row():
                        up = gr.Button("👍  Useful", size="sm")
                        down = gr.Button("👎  Not useful", size="sm")
                    comment = gr.Textbox(label="Comment (optional)", lines=1, placeholder="What was wrong or missing?", elem_classes="bs-plain")
                    footer = gr.HTML(elem_classes="bs-foot")

        results = [card, details, tiles, reason, found, steps, footer, request_id, comment]
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
        gr.HTML('<div class="bs-h">Quality over time</div><p class="bs-legend">Average of the last 10 requests.</p>')
        quality_plot = gr.LinePlot(x="time", y="value", color="signal", color_map=SIGNAL_COLOURS, height=280, y_lim=[0, 1],
                                   x_title="Time", y_title="Share", color_title="Signal", show_label=False, visible=False)
        quality_empty = gr.HTML(_empty("The chart appears after the first request."))
        gr.HTML(f'<div class="bs-h">Drift</div><p class="bs-legend">The last {monitor.WINDOW} requests against the reference set.</p>')
        drift_table = gr.HTML(_empty("Loading…"))
        gr.HTML('<div class="bs-h">Recent requests</div><p class="bs-legend">Summaries only, never the tender text.</p>')
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
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), theme=THEME, css=CSS, head=HEAD,
                allowed_paths=[str(FONT_DIR)])
