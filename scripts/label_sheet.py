"""
The sheet for the team's own labels, and its way back to eval/human_labels.csv.

    uv run --with openpyxl python scripts/label_sheet.py            # build eval/labelling_sheet.xlsx
    uv run --with openpyxl python scripts/label_sheet.py --write    # eval/human_labels.csv from the filled sheet

One row per case of eval/cases.csv: the tender text the app read (personal data already
removed), the app's recommendation, its reason and the facts it extracted, and two empty
columns: `label` (1 = the reasons match the tender, 0 = they do not) and `comment`.
The question is the judge's (bidsense/judge.py). The judge's own labels are NOT in the sheet,
so the team labels without seeing them; `main.py evaluate` then compares the two.

Building the sheet makes no Gemini call: it reads the app's answers from cache/ (run
scripts/warm_cache.py first). Building again keeps the labels already typed, but only while the
app's answer for that case is unchanged.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpyxl import Workbook, load_workbook  # noqa: E402
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402

import config  # noqa: E402
import pipeline  # noqa: E402
from bidsense import cache, decision  # noqa: E402
from bidsense.redact import redact  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHEET_PATH = ROOT / "eval" / "labelling_sheet.xlsx"
LABELS_PATH = ROOT / "eval" / "human_labels.csv"
SHEET = "Labels"
COLUMNS = [("id", 11), ("kind", 12), ("tender_text", 90), ("app_recommendation", 16), ("app_reason", 50),
           ("app_extracted_facts", 60), ("label", 8), ("comment", 45)]
FONT = "Arial"
CELL_LIMIT = 32000   # Excel holds 32,767 characters in a cell


def cases() -> list[dict]:
    with (ROOT / "eval" / "cases.csv").open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if (row.get("input") or "").strip()]


def answer(case: dict) -> dict | None:
    """The app's answer for a case, from the cache only (None if the case has not been read yet)."""
    text = redact(pipeline.read_input(case["input"]))[0][: config.MAX_INPUT_CHARS]
    cached = cache.get("agent", cache.key(config.GEMINI_MODEL, config.GEMINI_THINKING, config.PROMPT_VERSION, text))
    if not cached or cached.get("model") != config.GEMINI_MODEL:
        return None
    result = pipeline.run(case["input"], default_model_only=True)
    agent_output = result["outputs"]["agent"]
    facts = []
    for name, field in (agent_output.get("fields") or {}).items():
        if field.get("value") in (None, ""):
            facts.append(f"{name}: (not found)")
        else:
            quote = f' | evidence: "{field["evidence"]}"' if field.get("evidence") else ""
            facts.append(f"{name}: {field['value']} | confidence {field['confidence']:.2f}{quote}")
    if not agent_output.get("is_tender", True):   # out of scope: nothing was extracted, so there are no facts or confidences to show
        facts = ["The app says this is not a tender: " + str(agent_output.get("not_tender_reason", ""))]
    return {"text": text, "recommendation": decision.LABELS.get(result["decision"], result["decision"]),
            "reason": result["outputs"]["decision"]["reason"], "facts": "\n".join(facts)}


def typed_before() -> dict[str, dict]:
    if not SHEET_PATH.exists():
        return {}
    sheet = load_workbook(SHEET_PATH)[SHEET]
    header = [cell.value for cell in sheet[1]]
    return {str(row["id"]): row for row in (dict(zip(header, values)) for values in sheet.iter_rows(min_row=2, values_only=True)) if row.get("id")}


def build() -> None:
    kept, rows, missing = typed_before(), [], []
    for case in cases():
        found = answer(case)
        if found is None:
            missing.append(case["id"])
            continue
        before = kept.get(case["id"], {})
        same = before.get("app_reason") == found["reason"] and before.get("app_recommendation") == found["recommendation"]
        rows.append({"id": case["id"], "kind": case["kind"], "tender_text": found["text"][:CELL_LIMIT], "app_recommendation": found["recommendation"],
                     "app_reason": found["reason"], "app_extracted_facts": found["facts"][:CELL_LIMIT],
                     "label": before.get("label") if same else None, "comment": before.get("comment") if same else None})
    book = Workbook()
    sheet = book.active
    sheet.title = SHEET
    names = [name for name, _ in COLUMNS]
    thin = Side(style="thin", color="D0D0D0")
    for column, (name, width) in enumerate(COLUMNS, 1):
        cell = sheet.cell(row=1, column=column, value=name)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF", size=10)
        cell.fill = PatternFill("solid", fgColor="B45309" if name in ("label", "comment") else "1F2937")
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.row_dimensions[1].height = 28
    for index, row in enumerate(rows, 2):
        for column, name in enumerate(names, 1):
            cell = sheet.cell(row=index, column=column, value=row[name])
            cell.font = Font(name=FONT, size=10, bold=name == "app_recommendation")
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = Border(bottom=thin)
            if name in ("label", "comment"):
                cell.fill = PatternFill("solid", fgColor="FFF7CC")
        lines = max(sum(max(1, math.ceil(len(line) / 100)) for line in row["tender_text"].split("\n")),
                    sum(max(1, math.ceil(len(line) / 66)) for line in row["app_extracted_facts"].split("\n")))
        sheet.row_dimensions[index].height = min(400, max(60, 13.5 * lines + 4))
    letter = get_column_letter(names.index("label") + 1)
    check = DataValidation(type="list", formula1='"1,0"', allow_blank=True, showErrorMessage=True, errorTitle="label",
                           error="Type 1 (the reasons match the tender) or 0 (they do not).")
    check.add(f"{letter}2:{letter}{len(rows) + 1}")
    sheet.add_data_validation(check)
    sheet.freeze_panes = "C2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{len(rows) + 1}"

    guide = book.create_sheet("Read me")
    guide.column_dimensions["A"].width = 24
    guide.column_dimensions["B"].width = 115
    lines = [
        ("BidSense: your labels", ""),
        ("", ""),
        ("What to do", f"For at least 20 of the {len(rows)} rows of sheet '{SHEET}', read the tender text and what the app answered, and type "
                       "1 or 0 in the orange `label` column. `comment` is optional; one line on why helps most on the 0s."),
        ("The question", "Do the app's reasons match the tender? It is the same question the LLM judge answers, so the two can be compared."),
        ("label 1 (match)", "Every fact the reason relies on (contract value, what is bought, type of contract, whether it is a tender at "
                            "all) is supported by the text or is a sensible reading of it; the recommendation follows from the stated "
                            "reason (for example a review because a key field is missing, when it really is missing); and if the app "
                            "says the text is not a tender, that is true."),
        ("label 0 (mismatch)", "A fact is wrong or invented, a field is reported missing although the text states it clearly, or the text "
                               "is misclassified as a tender / not a tender."),
        ("Do not judge", "Whether bidding is a good business decision, or the predicted number of bidders: neither can be checked from "
                         "the text. A wrong recommendation with reasons that match the text is still a 1."),
        ("Not in this sheet", "The judge's labels and the expected answers of eval/cases.csv, on purpose: label before looking at them."),
        ("tender_text", "What the app read: personal data already replaced by tags such as [NAME], and long pages cut at "
                        f"{config.MAX_INPUT_CHARS:,} characters."),
        ("app_extracted_facts", "Each field the agent extracted, its confidence, and the quote it gave as evidence."),
        ("Example", "label = 0 | comment = says the value is missing, but the text gives 1,36 M EUR with VAT (illustration only)"),
        ("When you are done", "Save as eval/labelling_sheet.xlsx. `uv run --with openpyxl python scripts/label_sheet.py --write` copies the "
                              "labels to eval/human_labels.csv, and `uv run python main.py evaluate` reports the agreement with the judge."),
        ("Built with", f"model {config.GEMINI_MODEL}, prompt {config.PROMPT_VERSION}, bidders model {config.MODEL_VERSION}, bid cost EUR "
                       f"{config.DEFAULT_BID_COST_EUR:,.0f}, margin {config.DEFAULT_MARGIN_PCT:g}%."),
    ]
    for index, (label, text) in enumerate(lines, 1):
        a, b = guide.cell(row=index, column=1, value=label), guide.cell(row=index, column=2, value=text)
        a.font = Font(name=FONT, bold=True, size=14 if index == 1 else 10)
        b.font = Font(name=FONT, size=10)
        a.alignment = Alignment(vertical="top")
        b.alignment = Alignment(wrap_text=True, vertical="top")
    book.save(SHEET_PATH)
    print(f"{SHEET_PATH.relative_to(ROOT)}: {len(rows)} cases.")
    if missing:
        print(f"  Not in the sheet, because the app has not read them yet (run scripts/warm_cache.py): {', '.join(missing)}")


def write() -> None:
    known = {case["id"] for case in cases()}
    sheet = load_workbook(SHEET_PATH, data_only=True)[SHEET]
    header = [cell.value for cell in sheet[1]]
    labels, errors = [], []
    for values in sheet.iter_rows(min_row=2, values_only=True):
        row = dict(zip(header, values))
        typed = "" if row.get("label") is None else str(row["label"]).strip().split(".")[0]
        if typed == "":
            continue
        if typed not in ("0", "1"):
            errors.append(f"{row.get('id')}: label is '{row.get('label')}', not 1 or 0")
        if str(row.get("id")) not in known:
            errors.append(f"{row.get('id')}: no such case in eval/cases.csv")
        labels.append({"id": str(row.get("id")), "label": typed, "comment": str(row.get("comment") or "").strip()})
    for line in errors:
        print("  ERROR:", line)
    if errors:
        raise SystemExit("Fix the sheet and run again. Nothing was written.")
    with LABELS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "comment"])
        writer.writeheader()
        writer.writerows(labels)
    ones = sum(label["label"] == "1" for label in labels)
    print(f"Wrote {len(labels)} labels to {LABELS_PATH.relative_to(ROOT)} ({ones} ones, {len(labels) - ones} zeros).")
    if len(labels) < 20:
        print(f"  warning: the course asks for 20 or more labels; there are {len(labels)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="write eval/human_labels.csv from the filled sheet")
    write() if parser.parse_args().write else build()
