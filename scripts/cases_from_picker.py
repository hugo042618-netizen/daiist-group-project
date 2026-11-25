"""
Turn the sheet the team filled in (eval/case_picker.xlsx) into eval/cases.csv.

    uv run --with openpyxl python scripts/cases_from_picker.py            # check only, writes nothing
    uv run --with openpyxl python scripts/cases_from_picker.py --write    # write eval/cases.csv

A row is a test case when its `expected` cell is filled. Nothing is decided here: kind,
expected, baseline and notes are copied as the team typed them. The script only checks them
(allowed words, no empty baseline, no text used twice or already used by eval/reference.csv or
eval/shifted.csv, the course's minimum counts) and numbers the cases c01, c02...
The TED publication number is appended to the notes, so each case can be traced back.
After writing: scripts/fetch_candidates.py --annotate, scripts/warm_cache.py, main.py evaluate, check.py.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpyxl import load_workbook  # noqa: E402

from bidsense.redact import redact  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KINDS = ["normal", "hard", "review", "out_of_scope"]
DECISIONS = ["BID", "DONT_BID", "REVIEW", "OUT_OF_SCOPE"]
MINIMUM = {"hard": 5, "review": 5, "out_of_scope": 3, "normal": 1}
COLUMNS = ["id", "input", "kind", "expected", "baseline", "notes"]


def squash(text) -> str:
    return " ".join(str(text or "").split())


def clean(value) -> str:
    return str(value).strip() if value is not None else ""


def read_sheet(path: Path) -> list[dict]:
    sheet = load_workbook(path, data_only=True)["Candidates"]
    header = [clean(cell.value) for cell in sheet[1]]
    return [dict(zip(header, values)) | {"_row": number} for number, values in enumerate(sheet.iter_rows(min_row=2, values_only=True), 2)]


def convert(path: Path) -> tuple[list[dict], list[str], list[str]]:
    """(cases, errors, warnings). Errors must be fixed in the sheet before anything is written."""
    used = {}
    for name in ("reference.csv", "shifted.csv"):
        with (ROOT / "eval" / name).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                used[squash(row.get("input"))] = name
    cases, errors, warnings, seen = [], [], [], {}
    for row in read_sheet(path):
        expected = clean(row.get("expected")).upper().replace("'", "").replace(" ", "_")
        where = f"row {row['_row']} ({clean(row.get('id')) or 'no id'})"
        if not expected:
            if any(clean(row.get(name)) for name in ("kind", "baseline")):
                warnings.append(f"{where}: kind or baseline is filled but `expected` is empty, so the row is ignored")
            continue
        kind, baseline, text = clean(row.get("kind")).lower(), clean(row.get("baseline")), clean(row.get("text"))
        if expected not in DECISIONS:
            errors.append(f"{where}: expected is '{row.get('expected')}', not one of {', '.join(DECISIONS)}")
        if kind not in KINDS:
            errors.append(f"{where}: kind is '{row.get('kind')}', not one of {', '.join(KINDS)}")
        if not baseline:
            errors.append(f"{where}: baseline is empty")
        if not text:
            errors.append(f"{where}: text is empty")
        if squash(text) in used:
            errors.append(f"{where}: this text is already in eval/{used[squash(text)]}")
        if squash(text) in seen:
            errors.append(f"{where}: same text as {seen[squash(text)]}")
        seen[squash(text)] = where
        redacted, removed = redact(text)
        if removed:
            warnings.append(f"{where}: the personal-data filter would remove {removed} item(s) from this text; check it holds no personal data")
        notes, ted_id = clean(row.get("notes")), clean(row.get("id"))
        if ted_id and clean(row.get("url")):
            notes = f"{notes} (TED {ted_id})".strip()
        cases.append({"input": text, "kind": kind, "expected": expected, "baseline": baseline, "notes": notes})
    for number, case in enumerate(cases, 1):
        case["id"] = f"c{number:02d}"
    counts = {kind: sum(case["kind"] == kind for case in cases) for kind in KINDS}
    if len(cases) < 30:
        warnings.append(f"{len(cases)} cases: the course asks for 30 or more")
    for kind, least in MINIMUM.items():
        if counts[kind] < least:
            warnings.append(f"{counts[kind]} `{kind}` cases: the course asks for at least {least}" if kind != "normal"
                            else "no `normal` cases: the drift demo needs some")
    return cases, errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", type=Path, default=ROOT / "eval" / "case_picker.xlsx")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "cases.csv")
    parser.add_argument("--write", action="store_true", help="write the cases (without it, only the checks run)")
    args = parser.parse_args()
    cases, errors, warnings = convert(args.sheet)
    counts = {kind: sum(case["kind"] == kind for case in cases) for kind in KINDS}
    answers = {word: sum(case["expected"] == word for case in cases) for word in DECISIONS}
    print(f"{len(cases)} cases. By kind: " + ", ".join(f"{k} {n}" for k, n in counts.items())
          + ". By expected answer: " + ", ".join(f"{k} {n}" for k, n in answers.items()) + ".")
    for line in warnings:
        print("  warning:", line)
    for line in errors:
        print("  ERROR:", line)
    if errors:
        raise SystemExit("Fix the errors in the sheet and run again. Nothing was written.")
    if not args.write:
        print("Checks only. Run again with --write to write the file.")
        return
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(cases)
    print(f"Wrote {len(cases)} cases to {args.out}.")


if __name__ == "__main__":
    main()
