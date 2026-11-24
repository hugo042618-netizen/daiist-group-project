"""
The three tools the agent can call. They only read small files in data/: no network.

- lookup_cpv(query)                      top 5 codes of the official CPV 2008 list
- parse_amount(text)                     an amount written in words/digits -> euros as a number
- market_stats(cpv_division, contract_type)   how many bidders similar tenders had in our training data
"""

import csv
import json
import re
import unicodedata
from functools import lru_cache

import config
from bidsense import features

CPV_CSV = config.DATA_DIR / "cpv_2008.csv"
MARKET_STATS = config.DATA_DIR / "market_stats.json"


def _plain(text: str) -> str:
    """Lower case, without accents."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(text).lower()) if not unicodedata.combining(c))


# --- lookup_cpv -----------------------------------------------------------------

@lru_cache(maxsize=1)
def _cpv_index():
    from sklearn.feature_extraction.text import TfidfVectorizer

    with CPV_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Character n-grams cope with plurals and word endings in Spanish and English alike.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    matrix = vectorizer.fit_transform(_plain(f"{r['es']} | {r['en']}") for r in rows)
    return rows, vectorizer, matrix


@lru_cache(maxsize=1)
def cpv_codes() -> dict[str, dict]:
    """Every CPV code -> its row (code, es, en)."""
    return {r["code"]: r for r in _cpv_index()[0]}


def lookup_cpv(query: str) -> dict:
    """Top 5 CPV 2008 codes for a short description of what is being bought."""
    rows, vectorizer, matrix = _cpv_index()
    query = str(query or "").strip()
    if not query:
        return {"error": "empty query"}
    digits = re.sub(r"\D", "", query)
    if len(digits) >= 2 and len(digits) >= len(query.replace(" ", "").replace("-", "")) - 1:
        matches = [r for r in rows if r["code"].startswith(digits[:8])][:5]   # the query is a code: explain it
    else:
        scores = (matrix @ vectorizer.transform([_plain(query)]).T).toarray().ravel()
        matches = [rows[i] for i in scores.argsort()[::-1][:5] if scores[i] > 0]
    return {"results": [{"code": r["code"], "division": r["code"][:2], "es": r["es"], "en": r["en"]} for r in matches]}


# --- parse_amount ---------------------------------------------------------------

_MULTIPLIERS = [
    (r"miles de millones|billions?|mil millones|bn\b", 1e9),
    (r"millones|mill[oó]n|millions?|milh[oõ]es|milh[aã]o|mio\b|mm\b|m€|m\b", 1e6),
    (r"miles|mil\b|thousands?|k€|k\b", 1e3),
]
_NUMBER = re.compile(r"\d[\d.,'  ]*\d|\d")


def _to_float(number: str) -> float | None:
    """'1.234.567,89' (Spanish) or '1,234,567.89' (English) -> 1234567.89."""
    number = re.sub(r"[  ']", "", number).strip(".,")
    if not number:
        return None
    last_dot, last_comma = number.rfind("."), number.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        decimal = "." if last_dot > last_comma else ","
    elif last_dot >= 0 or last_comma >= 0:
        separator = "." if last_dot >= 0 else ","
        tail = number.split(separator)[-1]
        # "1.234" or "12,345,678" are thousands; "1,5" or "12.50" are decimals.
        decimal = separator if (len(tail) != 3 or number.count(separator) == 1 and len(number.split(separator)[0]) > 3) else None
        if number.count(separator) > 1:
            decimal = None
    else:
        decimal = None
    thousands = {".": ",", ",": ".", None: ".,"}[decimal]
    for character in thousands:
        number = number.replace(character, "")
    if decimal == ",":
        number = number.replace(",", ".")
    try:
        return float(number)
    except ValueError:
        return None


def parse_amount(text: str) -> dict:
    """An amount as written in a tender ("1.250.000,00 €", "2,5 millones de euros", "EUR 350k") -> euros."""
    text = str(text or "")
    match = _NUMBER.search(text)
    if not match:
        return {"error": "no number found", "text": text[:80]}
    value = _to_float(match.group())
    if value is None:
        return {"error": "could not read the number", "text": text[:80]}
    after = _plain(text[match.end(): match.end() + 25])
    for pattern, multiplier in _MULTIPLIERS:
        if re.match(rf"\s*(?:de\s+)?(?:{pattern})", after):
            value *= multiplier
            break
    currency = "EUR"
    if re.search(r"(?i)\b(usd|d[oó]lares|dollars)\b|\$", text):
        currency = "USD"
    elif re.search(r"(?i)\b(gbp|libras|pounds)\b|£", text):
        currency = "GBP"
    result = {"amount": round(value, 2), "currency": currency}
    if currency != "EUR":
        result["warning"] = "not in euros: do not use as estimated_value_eur without converting"
    return result


# --- market_stats -----------------------------------------------------------------

@lru_cache(maxsize=1)
def _market() -> dict:
    return json.loads(MARKET_STATS.read_text(encoding="utf-8"))


def market_stats(cpv_division: str, contract_type: str = "") -> dict:
    """Median and quartiles of the number of bidders in Spanish tenders of this CPV division (and contract type)."""
    division = features.cpv_division(str(cpv_division).zfill(2))
    table = _market()
    if division is None:
        return {"error": f"'{cpv_division}' is not a CPV division", "all_tenders": table["all"]}
    contract_type = str(contract_type or "").strip().lower()
    for key, scope in ((f"{division}|{contract_type}", f"CPV division {division}, {contract_type}"), (division, f"CPV division {division}")):
        if key in table:
            return {"scope": scope, "source": table.get("source", "Spanish award notices in TED"), "bidders": table[key]}
    return {"scope": "all Spanish tenders (too few in this division)", "source": table.get("source", "Spanish award notices in TED"),
            "bidders": table["all"]}
