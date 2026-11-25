"""
Build eval/case_picker.xlsx: the sheet the team fills in to choose the test cases.

    uv run --with openpyxl python scripts/build_case_picker.py

One row per candidate of eval/candidates/ that no eval/ file uses yet (`used_in` empty): the
text, TED's official fields and the real number of offers, four empty columns for the team
(kind, expected, baseline, notes) and a helper column with what the decision rule
(bidsense/decision.py, thresholds of config.py) gives when it is told the REAL number of
bidders and the official fields. The helper is not the expected answer: the team decides it.

Run again after changing a threshold in config.py: what the team already typed is kept
(matched by notice id). `scripts/cases_from_picker.py` turns the filled sheet into eval/cases.csv.
openpyxl is only needed by these two scripts, not by the app.
"""

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
from bidsense import decision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "case_picker.xlsx"
SHEET = "Candidates"
GROUP_ORDER = ["typical_fresh", "unusual", "very_large", "not_a_tender"]
KINDS = ["normal", "hard", "review", "out_of_scope"]
DECISIONS = [decision.BID, decision.DONT_BID, decision.REVIEW, decision.OUT_OF_SCOPE]
TEAM_COLUMNS = ["kind", "expected", "baseline", "notes"]
# (header, width, source): the source is a candidate column, a team column, or a computed value.
COLUMNS = [
    ("id", 13), ("group", 13), ("text", 95), ("estimated_value_eur", 15), ("real_offers", 9),
    ("rule_with_real_bidders", 16), ("win_chance", 9), ("expected_value_eur", 14),
    ("kind", 13), ("expected", 15), ("baseline", 15), ("notes", 40),
    ("cpv_code", 11), ("cpv_label", 38), ("contract_type", 11), ("procedure", 11), ("duration_months", 9),
    ("lots_number", 7), ("price_weight_pct", 9), ("is_framework_agreement", 10), ("region", 8), ("buyer_type", 19),
    ("notice_type", 13), ("published", 11), ("flags", 28), ("url", 46),
]
FONT = "Arial"
THIN = Side(style="thin", color="D0D0D0")


def number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def rule(row: dict) -> dict:
    """What the decision rule gives for this notice with the official fields and the real number of offers."""
    if row["group"] == "not_a_tender":
        return {"rule": decision.OUT_OF_SCOPE, "win": None, "ev": None}
    offers = number(row.get("offers"))
    fields = {"cpv_code": row.get("cpv_code") or None, "contract_type": row.get("contract_type") or None,
              "estimated_value_eur": number(row.get("estimated_value_eur"))}
    if offers is None:
        return {"rule": "", "win": None, "ev": None}
    outcome = decision.decide(fields, 1.0, offers, config.DEFAULT_BID_COST_EUR, config.DEFAULT_MARGIN_PCT)
    return {"rule": outcome["decision"], "win": outcome["win_chance"], "ev": outcome["expected_value_eur"]}


def candidates() -> list[dict]:
    squash = lambda text: " ".join(str(text).split())  # noqa: E731
    with (ROOT / "data" / "cpv_2008.csv").open(newline="", encoding="utf-8") as f:
        cpv = {r["code"]: r for r in csv.DictReader(f)}
    label_column = next((c for c in ("es", "label_es", "spa", "en", "label_en") if cpv and c in next(iter(cpv.values()))), None)
    examples = {squash(e["text"]) for e in json.loads((ROOT / "data" / "examples.json").read_text(encoding="utf-8"))}
    rows, seen, used_texts = [], set(), set()
    files = {path.stem: list(csv.DictReader(path.open(newline="", encoding="utf-8")))
             for path in sorted((ROOT / "eval" / "candidates").glob("*.csv"))}
    for group_rows in files.values():
        used_texts |= {squash(r["text"]) for r in group_rows if r.get("used_in")}
    for group in GROUP_ORDER + sorted(set(files) - set(GROUP_ORDER)):
        for row in files.get(group, []):
            text = squash(row["text"])
            if row.get("used_in") or text in used_texts or text in seen:
                continue   # already used by an eval/ file, or the same text as another candidate
            seen.add(text)
            code = str(row.get("cpv_code") or "").split(".")[0].zfill(8) if row.get("cpv_code") else ""
            flags = []
            if text in examples:
                flags.append("also an example button in the app")
            if row["group"] != "not_a_tender" and number(row.get("offers")) is None:
                flags.append("no real number of offers")
            rows.append(row | rule(row) | {"cpv_code": code, "cpv_label": (cpv.get(code) or {}).get(label_column, "") if label_column else "",
                                           "flags": "; ".join(flags)})
    return rows


def typed_before() -> dict[str, dict]:
    """What the team already typed in an existing sheet, by notice id."""
    if not OUT.exists():
        return {}
    sheet = load_workbook(OUT)[SHEET]
    header = [cell.value for cell in sheet[1]]
    kept = {}
    for values in sheet.iter_rows(min_row=2, values_only=True):
        row = dict(zip(header, values))
        typed = {name: row.get(name) for name in TEAM_COLUMNS if row.get(name) not in (None, "")}
        if typed and row.get("id"):
            kept[str(row["id"])] = typed
    return kept


def build() -> tuple[int, dict]:
    rows, kept = candidates(), typed_before()
    book = Workbook()
    sheet = book.active
    sheet.title = SHEET
    names = [name for name, _ in COLUMNS]
    header_fill, team_fill, helper_fill = PatternFill("solid", fgColor="1F2937"), PatternFill("solid", fgColor="FFF7CC"), PatternFill("solid", fgColor="E8F1FB")
    for column, (name, width) in enumerate(COLUMNS, 1):
        cell = sheet.cell(row=1, column=column, value=name)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF", size=10)
        cell.fill = PatternFill("solid", fgColor="B45309") if name in TEAM_COLUMNS else header_fill
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.row_dimensions[1].height = 32

    numbers = {"estimated_value_eur": "#,##0", "expected_value_eur": "#,##0;[Red]-#,##0", "win_chance": "0%", "real_offers": "0.#",
               "duration_months": "0.#", "lots_number": "0", "price_weight_pct": "0.#"}
    for index, row in enumerate(rows, 2):
        values = {
            "id": row["id"], "group": row["group"], "text": row["text"], "estimated_value_eur": number(row.get("estimated_value_eur")),
            "real_offers": number(row.get("offers")), "rule_with_real_bidders": row["rule"], "win_chance": row["win"],
            "expected_value_eur": row["ev"], "cpv_code": row["cpv_code"], "cpv_label": row["cpv_label"],
            "contract_type": row.get("contract_type", ""), "procedure": row.get("procedure", ""),
            "duration_months": number(row.get("duration_months")), "lots_number": number(row.get("lots_number")),
            "price_weight_pct": number(row.get("price_weight_pct")), "is_framework_agreement": row.get("is_framework_agreement", ""),
            "region": row.get("region", ""), "buyer_type": row.get("buyer_type", ""), "notice_type": row.get("notice_type", ""),
            "published": row.get("published", ""), "flags": row["flags"], "url": row.get("url", ""),
        } | kept.get(str(row["id"]), {})
        for column, name in enumerate(names, 1):
            cell = sheet.cell(row=index, column=column, value=values.get(name))
            cell.font = Font(name=FONT, size=10, bold=name == "rule_with_real_bidders")
            cell.alignment = Alignment(wrap_text=name in ("text", "notes", "cpv_label", "flags"), vertical="top")
            cell.border = Border(bottom=THIN)
            if name in numbers:
                cell.number_format = numbers[name]
            if name in TEAM_COLUMNS:
                cell.fill = team_fill
            elif name in ("rule_with_real_bidders", "win_chance", "expected_value_eur"):
                cell.fill = helper_fill
            if name == "url" and values.get(name):
                cell.hyperlink = values[name]
                cell.font = Font(name=FONT, size=10, color="0563C1", underline="single")
        lines = sum(max(1, math.ceil(len(line) / 105)) for line in str(row["text"]).split("\n"))
        sheet.row_dimensions[index].height = min(400, max(30, 13.5 * lines + 4))

    last = len(rows) + 1
    for name, allowed in (("kind", KINDS), ("expected", DECISIONS), ("baseline", DECISIONS)):
        letter = get_column_letter(names.index(name) + 1)
        check = DataValidation(type="list", formula1='"' + ",".join(allowed) + '"', allow_blank=True, showErrorMessage=True,
                               errorTitle=name, error="Choose one of: " + ", ".join(allowed))
        check.add(f"{letter}2:{letter}{last + 60}")
        sheet.add_data_validation(check)
    sheet.freeze_panes = "C2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{last}"

    legend(book, rows)
    book.save(OUT)
    counts = {}
    for row in rows:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    return len(rows), counts


def legend(book: Workbook, rows: list[dict]) -> None:
    sheet = book.create_sheet("Read me")
    sheet.column_dimensions["A"].width = 26
    sheet.column_dimensions["B"].width = 110
    lines = [
        ("BidSense: picking the test cases", ""),
        ("", ""),
        ("What this is", f"{len(rows)} real TED notices that no eval/ file uses yet (sheet '{SHEET}'), one per row. "
                         "Nothing here is a test case until the team fills it in."),
        ("What to fill in", "Only the four orange columns: kind, expected, baseline, notes. A row becomes a test case when "
                            "'expected' is filled; rows left empty are ignored."),
        ("kind", "normal, hard (should work but is difficult), review (should go to a person) or out_of_scope. The course asks "
                 "for 30+ cases with at least 5 hard, 5 review and 3 out_of_scope, and some normal ones (the drift demo uses them)."),
        ("expected", "BID, DONT_BID, REVIEW or OUT_OF_SCOPE: what a correct answer is. The team decides it."),
        ("baseline", "What the company would do with this tender today, without the product, in the same four words."),
        ("notes", "Optional: why the case is in the set."),
        ("", ""),
        ("rule_with_real_bidders", "Helper, not the answer: what the decision rule gives when it is told TED's official fields and the "
                                   "REAL number of offers, instead of the agent's reading and the model's prediction. Blank when the "
                                   "notice has no number of offers."),
        ("win_chance", "1 / real offers."),
        ("expected_value_eur", "win chance x estimated value x margin - bid cost."),
        ("Settings used", f"Bid cost EUR {config.DEFAULT_BID_COST_EUR:,.0f}; margin {config.DEFAULT_MARGIN_PCT:g}%; REVIEW when the value is above "
                          f"EUR {config.REVIEW_VALUE_EUR:,.0f} or the expected value is within EUR {config.CLOSE_CALL_EUR:,.0f} of zero "
                          "(config.py on the day the sheet was built). If a threshold changes, run scripts/build_case_picker.py again: "
                          "it recomputes the helper and keeps what you typed."),
        ("real_offers", "Tenders actually received (TED), the median across lots. The app never sees it."),
        ("group", "typical_fresh: open procedure, one lot, EUR 100k to 2M. unusual: restricted or negotiated procedures, frameworks, "
                  "many lots. very_large: above EUR 10M. not_a_tender: prior information notices and contract modifications."),
        ("Other columns", "TED's official fields for the notice. 'text' is what the app will read; the CPV code and the type of "
                          "contract are left out of it on purpose."),
        ("Your own cases", "Add rows at the bottom with your own text (no personal data), an id of your choice, and the orange columns."),
        ("", ""),
        ("Example of a filled row", "kind = normal | expected = BID | baseline = BID | notes = small services contract, few bidders "
                                    "(illustration only, not a real case)"),
        ("When you are done", "Save the file as eval/case_picker.xlsx. It is turned into eval/cases.csv with "
                              "scripts/cases_from_picker.py."),
    ]
    for index, (label, text) in enumerate(lines, 1):
        a, b = sheet.cell(row=index, column=1, value=label), sheet.cell(row=index, column=2, value=text)
        a.font = Font(name=FONT, bold=True, size=14 if index == 1 else 10)
        b.font = Font(name=FONT, size=10)
        a.alignment = Alignment(vertical="top")
        b.alignment = Alignment(wrap_text=True, vertical="top")


if __name__ == "__main__":
    total, per_group = build()
    print(f"{OUT.relative_to(ROOT)}: {total} candidates (" + ", ".join(f"{group} {n}" for group, n in per_group.items()) + ")")
