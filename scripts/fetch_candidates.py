"""
Collect CANDIDATES for the team's test files from real TED notices. `uv run python scripts/fetch_candidates.py`

The team writes eval/cases.csv and eval/human_labels.csv; this script only gathers real
material to choose from, in eval/candidates/:

    typical.csv         Spanish award notices: open procedure, one lot, mid-sized
    unusual.csv         Spanish award notices: restricted / negotiated procedures, frameworks, many lots
    very_large.csv      Spanish award notices above 10 million euros
    not_a_tender.csv    Spanish notices that are NOT calls for tenders (prior information, contract modifications)
    reference_pool.csv  normal Spanish notices, not in the files above -> eval/reference.csv
    shifted_pool.csv    Portuguese and French notices -> eval/shifted.csv

It also writes eval/extraction_truth.csv (the official fields of the tender candidates,
for the component-1 evaluation) and, with --fill, eval/reference.csv and eval/shifted.csv.

`--annotate` fetches no new notices: it only refreshes the `used_in` column (which eval/
file already uses each notice) and the official `buyer_type`. Run it after editing
eval/cases.csv, eval/reference.csv or eval/shifted.csv.

Source: the TED Search API (https://docs.ted.europa.eu/api/latest/search.html), no key needed.
Offline only: the app never calls it.

How each `text` is built: the API returns a notice as separate fields, so the text is the
notice's own title and description (verbatim) followed by the tender conditions as labelled
lines (value, duration, procedure, lots, award criteria, place). The CPV code and the type
of contract are left out on purpose, so the agent has to work them out. The results section
(winners, number of offers received) and all contact details are never included, and the
text goes through the same redaction as the app.
"""

import argparse
import csv
import json
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bidsense import features  # noqa: E402
from bidsense.redact import redact  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "candidates"
API = "https://api.ted.europa.eu/v3/notices/search"
SEED = 7
SINCE = "20260101"

FIELDS = [
    "publication-number", "publication-date", "notice-type", "official-language", "buyer-name",
    "title-proc", "description-proc", "title-lot", "description-lot", "BT-137-Lot",
    "main-classification-proc", "contract-nature-main-proc", "procedure-type",
    "estimated-value-proc", "estimated-value-cur-proc", "BT-36-Lot", "framework-agreement-lot",
    "place-of-performance-subdiv-proc", "place-of-performance-subdiv-lot",
    "award-criterion-type-lot", "award-criterion-number-lot", "award-criterion-number-weight-lot",
    "received-submissions-type-code", "received-submissions-type-val", "modification-description", "buyer-legal-type",
]
# eForms buyer legal type -> the agent's buyer_type (everything else is "other").
BUYER_TYPE = {"la": "city_council", "ra": "regional_government", "cga": "ministry"}
PROCEDURE = {"open": "open", "restricted": "restricted", "neg-w-call": "negotiated", "neg-wo-call": "negotiated",
             "comp-dial": "other", "innovation": "other", "comp-tend": "other", "oth-single": "other", "oth-mult": "other"}
LANGUAGE = {"SPA": ("spa", "es"), "POR": ("por", "pt"), "FRA": ("fra", "fr"), "ENG": ("eng", "en"), "CAT": ("cat", "ca")}

LABELS = {
    "es": {"title": "Título", "buyer": "Órgano de contratación", "description": "Descripción", "lots": "Lotes", "lot": "Lote",
           "value": "Valor estimado sin IVA", "duration": "Duración del contrato", "months": "meses",
           "procedure": "Tipo de procedimiento", "framework": "Acuerdo marco", "yes": "sí", "no": "no",
           "place": "Lugar de ejecución", "criteria": "Criterios de adjudicación", "price": "precio", "quality": "calidad",
           "open": "abierto", "restricted": "restringido", "negotiated": "negociado con publicidad", "other": "otro",
           "pin-only": "Tipo de anuncio: anuncio de información previa (no es una convocatoria de licitación)",
           "can-modif": "Tipo de anuncio: modificación de un contrato ya adjudicado", "modification": "Modificación"},
    "pt": {"title": "Título", "buyer": "Entidade adjudicante", "description": "Descrição", "lots": "Lotes", "lot": "Lote",
           "value": "Valor estimado sem IVA", "duration": "Duração do contrato", "months": "meses",
           "procedure": "Tipo de procedimento", "framework": "Acordo-quadro", "yes": "sim", "no": "não",
           "place": "Local de execução", "criteria": "Critérios de adjudicação", "price": "preço", "quality": "qualidade",
           "open": "concurso público", "restricted": "concurso limitado", "negotiated": "procedimento de negociação", "other": "outro"},
    "fr": {"title": "Titre", "buyer": "Acheteur", "description": "Description", "lots": "Lots", "lot": "Lot",
           "value": "Valeur estimée hors TVA", "duration": "Durée du marché", "months": "mois",
           "procedure": "Type de procédure", "framework": "Accord-cadre", "yes": "oui", "no": "non",
           "place": "Lieu d'exécution", "criteria": "Critères d'attribution", "price": "prix", "quality": "qualité",
           "open": "procédure ouverte", "restricted": "procédure restreinte", "negotiated": "procédure négociée", "other": "autre"},
}
LABELS["en"] = LABELS["ca"] = LABELS["es"]

GROUPS = {
    # name: (query, how many to keep, filter)
    "typical": ("buyer-country=ESP AND notice-type=can-standard AND procedure-type=open AND estimated-value-proc>100000 "
                "AND estimated-value-proc<2000000", 40),
    "unusual": ("buyer-country=ESP AND notice-type=can-standard AND estimated-value-proc>50000 AND "
                "(procedure-type=restricted OR procedure-type=neg-w-call OR procedure-type=comp-dial OR framework-agreement-lot=fa-wo-rc "
                "OR framework-agreement-lot=fa-w-rc OR framework-agreement-lot=fa-mix)", 25),
    "very_large": ("buyer-country=ESP AND notice-type=can-standard AND estimated-value-proc>10000000", 15),
    "not_a_tender": ("buyer-country=ESP AND (notice-type=pin-only OR notice-type=can-modif)", 20),
    # Same definition of "normal" as `typical`, different notices: the drift check compares live inputs with these.
    "reference_pool": ("buyer-country=ESP AND notice-type=can-standard AND procedure-type=open AND estimated-value-proc>100000 "
                       "AND estimated-value-proc<2000000", 45),
    "shifted_pool": ("(buyer-country=PRT OR buyer-country=FRA) AND notice-type=can-standard AND estimated-value-proc>100000", 20),
}
COLUMNS = ["id", "group", "used_in", "notice_type", "published", "language", "text", "cpv_code", "contract_type", "estimated_value_eur",
           "duration_months", "procedure", "lots_number", "price_weight_pct", "is_framework_agreement", "region", "buyer_type",
           "offers", "url"]
TRUTH_COLUMNS = ["id", "input", "cpv_code", "contract_type", "estimated_value_eur", "duration_months", "procedure", "lots_number",
                 "price_weight_pct", "is_framework_agreement", "region", "buyer_type", "offers"]
USED_FILES = ("reference.csv", "shifted.csv", "cases.csv")   # eval/ files whose inputs must not be reused elsewhere


def search(query: str, page: int, limit: int = 100) -> list[dict]:
    body = json.dumps({"query": f"{query} AND publication-date>={SINCE}", "fields": FIELDS, "limit": limit, "page": page,
                       "scope": "ALL"}).encode()
    request = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})  # noqa: S310 - fixed https URL
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                return json.loads(response.read()).get("notices", [])
        except Exception as error:  # noqa: BLE001
            if attempt == 3:
                raise SystemExit(f"TED API request failed: {type(error).__name__}: {error}") from error
            time.sleep(5 * (attempt + 1))
    return []


def _text(value, language_key: str) -> str:
    """A multilingual field -> one string in the notice's language."""
    if isinstance(value, dict):
        value = value.get(language_key) or next(iter(value.values()), "")
    if isinstance(value, list):
        value = value[0] if value else ""
    return " ".join(str(value or "").split())


def _texts(value, language_key: str) -> list[str]:
    if isinstance(value, dict):
        value = value.get(language_key) or next(iter(value.values()), [])
    if not isinstance(value, list):
        value = [value]
    return [" ".join(str(v or "").split()) for v in value if v]


def _duration_months(periods) -> float | None:
    months = []
    for period in periods or []:
        try:
            amount = float(period["value"])
        except (KeyError, TypeError, ValueError):
            continue
        months.append(amount * {"YEAR": 12, "MONTH": 1, "WEEK": 7 / 30.4, "DAY": 1 / 30.4}.get(period.get("unit"), float("nan")))
    months = [m for m in months if m == m and m > 0]
    return round(statistics.median(months), 1) if months else None


def _price_weight(notice: dict, lots: int) -> float | None:
    """Share of price/cost in the award criteria, only when the notice has one lot and the weights add up."""
    kinds, numbers = notice.get("award-criterion-type-lot") or [], notice.get("award-criterion-number-lot") or []
    if lots != 1 or not kinds:
        return None
    if len(kinds) == 1 and kinds[0] in ("price", "cost"):
        return 100.0
    if len(kinds) != len(numbers):
        return None
    try:
        numbers = [float(n) for n in numbers]
    except (TypeError, ValueError):
        return None
    total = sum(numbers)
    if total <= 0:
        return None
    return round(100 * sum(n for kind, n in zip(kinds, numbers) if kind in ("price", "cost")) / total, 1)


def _offers(notice: dict) -> float | None:
    codes, values = notice.get("received-submissions-type-code") or [], notice.get("received-submissions-type-val") or []
    tenders = [float(v) for code, v in zip(codes, values) if code == "tenders" and str(v).replace(".", "").isdigit()]
    tenders = [t for t in tenders if t >= 1]
    return statistics.median(tenders) if tenders else None


def _money(value: float, language: str) -> str:
    text = f"{value:,.2f}"
    return (text.replace(",", " ").replace(".", ",") if language in ("es", "pt", "fr", "ca") else text) + " EUR"


def build(notice: dict, group: str) -> dict | None:
    """One candidate row from one notice, or None if the notice is not usable."""
    official = (notice.get("official-language") or ["SPA"])[0]
    if official not in LANGUAGE:
        return None
    language_key, language = LANGUAGE[official]
    labels = LABELS[language]
    title = _text(notice.get("title-proc"), language_key)
    description = _text(notice.get("description-proc"), language_key)
    if len(title) < 15 or len(title) + len(description) < 80:
        return None
    lot_ids = notice.get("BT-137-Lot") or []
    lots = max(1, len(set(lot_ids)))
    notice_type = notice.get("notice-type", "")
    lines = [f"{labels['title']}: {title}"]
    if buyer := _text(notice.get("buyer-name"), language_key):
        lines.append(f"{labels['buyer']}: {buyer}")
    if description and description != title:
        lines.append(f"{labels['description']}: {description}")

    legal_type = (notice.get("buyer-legal-type") or [""])[0]
    row = {"id": notice["publication-number"], "group": group, "notice_type": notice_type, "used_in": "",
           "buyer_type": BUYER_TYPE.get(legal_type, "other") if legal_type else "",
           "published": str(notice.get("publication-date", ""))[:10], "language": language,
           "url": f"https://ted.europa.eu/{language}/notice/-/detail/{notice['publication-number']}"}

    if group == "not_a_tender":
        lines.insert(0, LABELS["es"][notice_type])
        if modification := _text(notice.get("modification-description"), language_key):
            lines.append(f"{LABELS['es']['modification']}: {modification}")
    else:
        value = notice.get("estimated-value-proc")
        if notice.get("estimated-value-cur-proc") != "EUR" or value is None:
            return None
        value, offers = float(value), _offers(notice)
        if offers is None:
            return None
        cpv = notice.get("main-classification-proc") or ""
        cpv = str(cpv[0] if isinstance(cpv, list) else cpv)[:8]
        if features.cpv_division(cpv) is None:
            return None
        if lots > 1:
            lot_titles = _texts(notice.get("title-lot"), language_key)[:6]
            lines.append(f"{labels['lots']}: {lots}" + ("".join(f"\n- {labels['lot']} {i}: {t}" for i, t in enumerate(lot_titles, 1))))
        lines.append(f"{labels['value']}: {_money(value, language)}")
        duration = _duration_months(notice.get("BT-36-Lot"))
        if duration:
            lines.append(f"{labels['duration']}: {duration:g} {labels['months']}")
        procedure = PROCEDURE.get(notice.get("procedure-type"), "other")
        lines.append(f"{labels['procedure']}: {labels[procedure]}")
        framework = any(f not in ("none", None) for f in notice.get("framework-agreement-lot") or [])
        lines.append(f"{labels['framework']}: {labels['yes'] if framework else labels['no']}")
        nuts = (notice.get("place-of-performance-subdiv-proc") or notice.get("place-of-performance-subdiv-lot") or [""])[0]
        region = nuts[:4] if nuts[:4] in features.REGIONS else ("ES" if nuts.startswith("ES") else ("other" if nuts else ""))
        if nuts:
            lines.append(f"{labels['place']}: {nuts}" + (f" ({features.REGIONS[region]})" if region in features.REGIONS else ""))
        weight = _price_weight(notice, lots)
        if weight is not None:
            lines.append(f"{labels['criteria']}: {labels['price']} {weight:g} %, {labels['quality']} {100 - weight:g} %")
        row |= {"cpv_code": cpv, "contract_type": notice.get("contract-nature-main-proc", ""), "estimated_value_eur": value,
                "duration_months": duration, "procedure": procedure, "lots_number": lots, "price_weight_pct": weight,
                "is_framework_agreement": framework, "region": region, "offers": offers}

    text, _ = redact("\n".join(lines))
    return row | {"text": text}


def collect(group: str, query: str, wanted: int, taken: set[str], rng: random.Random) -> list[dict]:
    """Up to `wanted` candidates for a group: several pages, shuffled, at most 4 per CPV division for variety."""
    pool = []
    for page in rng.sample(range(1, 9), 4):
        pool += search(query, page)
    rng.shuffle(pool)
    rows, per_division, per_kind = [], {}, {}
    for notice in pool:
        if notice["publication-number"] in taken:
            continue
        row = build(notice, group)
        if row is None:
            continue
        division = str(row.get("cpv_code", ""))[:2]
        if group in ("typical", "reference_pool") and (row["lots_number"] != 1 or row["language"] != "es"):
            continue
        if group != "not_a_tender" and per_division.get(division, 0) >= (4 if group != "reference_pool" else 6):
            continue
        kind = row["language"] if group == "shifted_pool" else row["notice_type"]
        if group in ("shifted_pool", "not_a_tender") and per_kind.get(kind, 0) >= wanted // 2:
            continue
        if group == "shifted_pool" and row["language"] not in ("pt", "fr"):
            continue
        per_division[division] = per_division.get(division, 0) + 1
        per_kind[kind] = per_kind.get(kind, 0) + 1
        taken.add(row["id"])
        rows.append(row)
        if len(rows) == wanted:
            break
    return rows


def write(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def buyer_types(ids: list[str]) -> dict[str, str]:
    """The official buyer type of notices already collected, asked from the API by publication number."""
    found = {}
    for start in range(0, len(ids), 40):
        chunk = ids[start:start + 40]
        body = json.dumps({"query": "publication-number IN (" + " ".join(chunk) + ")", "limit": 100, "scope": "ALL",
                           "fields": ["publication-number", "buyer-legal-type"]}).encode()
        request = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})  # noqa: S310
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            for notice in json.loads(response.read()).get("notices", []):
                legal_type = (notice.get("buyer-legal-type") or [""])[0]
                found[notice["publication-number"]] = BUYER_TYPE.get(legal_type, "other") if legal_type else ""
    return found


def annotate(fetch_buyer_types: bool = True) -> None:
    """Update the candidate files in place, without changing which notices they hold or their texts.

    `used_in` says which eval/ file already uses a notice (reference.csv, shifted.csv, cases.csv), so that
    nobody reuses a reference input as a test case. `buyer_type` is TED's official buyer type.
    """
    squash = lambda text: " ".join(str(text).split())  # noqa: E731
    used = {}
    for name in USED_FILES:
        for row in read(ROOT / "eval" / name):
            used.setdefault(squash(row.get("input", "")), name)
    files = {path: read(path) for path in sorted(OUT.glob("*.csv"))}
    types = buyer_types([row["id"] for rows in files.values() for row in rows if not row.get("buyer_type")]) if fetch_buyer_types else {}
    for path, rows in files.items():
        for row in rows:
            row["used_in"] = used.get(squash(row["text"]), "")
            row["buyer_type"] = row.get("buyer_type") or types.get(row["id"], "")
        write(path, COLUMNS, rows)
        marked = sum(bool(row["used_in"]) for row in rows)
        print(f"  {path.name}: {len(rows)} notices, {marked} already used" + (f" ({rows[0]['used_in']})" if marked == len(rows) and rows else ""))
    by_id = {row["id"]: row for rows in files.values() for row in rows}
    truth = read(ROOT / "eval" / "extraction_truth.csv")
    for row in truth:
        row["buyer_type"] = by_id.get(row["id"], {}).get("buyer_type", "")
    if truth:
        write(ROOT / "eval" / "extraction_truth.csv", TRUTH_COLUMNS, truth)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fill", action="store_true",
                        help="also write eval/reference.csv and eval/shifted.csv from the pools (overwrites them)")
    parser.add_argument("--annotate", action="store_true",
                        help="do not fetch new notices: only refresh `used_in` and `buyer_type` in the existing candidate files")
    args = parser.parse_args()
    if args.annotate:
        annotate()
        return

    rng, taken, groups = random.Random(SEED), set(), {}
    for group, (query, wanted) in GROUPS.items():
        print(f"{group}: searching TED...", flush=True)
        groups[group] = collect(group, query, wanted, taken, rng)
        write(OUT / f"{group}.csv", COLUMNS, groups[group])
        print(f"  {len(groups[group])} candidates -> eval/candidates/{group}.csv")

    truth = [row | {"input": row["text"]} for group in ("typical", "unusual", "very_large") for row in groups[group]]
    write(ROOT / "eval" / "extraction_truth.csv", TRUTH_COLUMNS, truth)
    print(f"eval/extraction_truth.csv: official fields of {len(truth)} tender candidates.")

    if args.fill:
        write(ROOT / "eval" / "reference.csv", ["input"], [{"input": r["text"]} for r in groups["reference_pool"]])
        write(ROOT / "eval" / "shifted.csv", ["input"], [{"input": r["text"]} for r in groups["shifted_pool"]])
        print(f"eval/reference.csv: {len(groups['reference_pool'])} inputs. eval/shifted.csv: {len(groups['shifted_pool'])} inputs.")
    annotate(fetch_buyer_types=False)
    print("\neval/cases.csv and eval/human_labels.csv are the team's: pick and write them by hand (see eval/candidates/README.md).")


if __name__ == "__main__":
    main()
