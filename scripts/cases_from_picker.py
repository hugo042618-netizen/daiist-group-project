"""
Turn the sheet the team filled in (eval/case_picker.xlsx) into eval/cases.csv.

    uv run --with openpyxl python scripts/cases_from_picker.py            # check only, writes nothing
    uv run --with openpyxl python scripts/cases_from_picker.py --write    # write eval/cases.csv

A row is a test case when its `expected` cell is filled. Nothing is decided here: kind,
expected, baseline and notes are the team's. The script only tidies and checks them:

- `expected` may be typed as "2 bidders → DONT_BID": the label after the arrow is kept, and the
  bidder count goes to the notes if it is not there already;
- `--set-baseline out_of_scope=OUT_OF_SCOPE` replaces the baseline of every case of one kind (an
  instruction of the team, not a default);
- `text` may be a file under eval/inputs/ (the path is kept as the input);
- checks: allowed words, no empty baseline, no input used twice or already in eval/reference.csv or
  eval/shifted.csv, the course's minimum counts;
- ids: the team's own ids are kept (hard_01, oos_01...); TED candidates are numbered by kind
  (normal_01, review_01...) and their TED publication number is appended to the notes.
After writing: scripts/fetch_candidates.py --annotate, scripts/warm_cache.py, main.py evaluate, check.py.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpyxl import load_workbook  # noqa: E402

import config  # noqa: E402
import pipeline  # noqa: E402
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


def split_expected(typed: str) -> tuple[str, str]:
    """("DONT_BID", "2 bidders") from "2 bidders → DONT_BID"; a plain label gives (label, "")."""
    before, arrow, after = typed.rpartition("→") if "→" in typed else typed.rpartition("->")
    label = (after if arrow else typed).strip().upper().replace("'", "").replace(" ", "_")
    return label, before.strip() if arrow else ""


def convert(path: Path, set_baseline: dict[str, str] | None = None) -> tuple[list[dict], list[str], list[str]]:
    """(cases, errors, warnings). Errors must be fixed in the sheet before anything is written."""
    set_baseline = set_baseline or {}
    used = {}
    for name in ("reference.csv", "shifted.csv"):
        with (ROOT / "eval" / name).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                used[squash(row.get("input"))] = name
    cases, errors, warnings, seen = [], [], [], {}
    for row in read_sheet(path):
        expected, bidders = split_expected(clean(row.get("expected")))
        where = f"row {row['_row']} ({clean(row.get('id')) or 'no id'})"
        if not expected:
            if any(clean(row.get(name)) for name in ("kind", "baseline")):
                warnings.append(f"{where}: kind or baseline is filled but `expected` is empty, so the row is ignored")
            continue
        kind, baseline, text = clean(row.get("kind")).lower(), clean(row.get("baseline")), clean(row.get("text"))
        baseline = set_baseline.get(kind, baseline)
        content = pipeline.read_input(text)   # the file's text when `text` is a path under eval/inputs/
        from_file = content != text
        if expected not in DECISIONS:
            errors.append(f"{where}: expected is '{row.get('expected')}', not one of {', '.join(DECISIONS)}")
        if kind not in KINDS:
            errors.append(f"{where}: kind is '{row.get('kind')}', not one of {', '.join(KINDS)}")
        if not baseline:
            errors.append(f"{where}: baseline is empty")
        if not text:
            errors.append(f"{where}: text is empty")
        if baseline and baseline.upper() not in DECISIONS:
            warnings.append(f"{where}: baseline is '{baseline}', not one of the four decision words, so it cannot be scored")
        if squash(content) in used:
            errors.append(f"{where}: this text is already in eval/{used[squash(content)]}")
        if squash(content) in seen:
            errors.append(f"{where}: same text as {seen[squash(content)]}")
        seen[squash(content)] = where
        redacted, removed = redact(content)
        if removed:
            warnings.append(f"{where}: the personal-data filter removes {removed} from this text before the app reads it; check the input holds no personal data")
        if len(redacted) > config.MAX_INPUT_CHARS:
            warnings.append(f"{where}: {len(redacted):,} characters; the app reads the first {config.MAX_INPUT_CHARS:,}")
        notes, sheet_id = clean(row.get("notes")), clean(row.get("id"))
        if bidders and bidders.lower() not in notes.lower():
            notes = f"{notes} Real offers: {bidders}.".strip()
        ted = re.search(r"(\d+-20\d\d)", clean(row.get("url")) or "")
        if ted:
            notes = f"{notes} (TED {ted.group(1)})".strip()
        own_id = sheet_id if sheet_id and not re.fullmatch(r"\d+-20\d\d", sheet_id) else ""
        cases.append({"id": own_id, "input": text, "kind": kind, "expected": expected, "baseline": baseline.upper() if baseline.upper() in DECISIONS else baseline,
                      "notes": notes})
    taken, counters = {case["id"] for case in cases if case["id"]}, {}
    for case in cases:
        while not case["id"]:
            counters[case["kind"]] = counters.get(case["kind"], 0) + 1
            candidate = f"{case['kind'] or 'case'}_{counters[case['kind']]:02d}"
            if candidate not in taken:
                case["id"] = candidate
                taken.add(candidate)
    if len({case["id"] for case in cases}) != len(cases):
        errors.append("two cases have the same id")
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
    parser.add_argument("--set-baseline", action="append", default=[], metavar="KIND=WORD",
                        help="baseline for every case of one kind, e.g. out_of_scope=OUT_OF_SCOPE (the team's instruction)")
    args = parser.parse_args()
    cases, errors, warnings = convert(args.sheet, dict(item.split("=", 1) for item in args.set_baseline))
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
