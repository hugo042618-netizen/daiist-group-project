"""
Is an evidence quote really in the text? A check for display and for the evaluation only.

The agent gives a short quote for each value it extracts. `in_text` says whether that quote
appears in the text the agent read, ignoring upper and lower case and all white space (a quote
may join two lines, or lose a double space). It does not change confidences or decisions.

(Separately, and since the first version, bidsense/agent.py caps a field's confidence at 0.5
when its quote is not in the text; that happens when the agent's answer is validated, and it is
not touched here.)
"""

import re


def _bare(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def in_text(quote: str, text: str, exact: bool = False) -> bool | None:
    """True or False for a quote; None when there is no quote to check.

    The agent sometimes shortens a quote with dots ("Título: Suministro..."): the pieces around the dots
    must all be in the text, in that order. With `exact`, the dots are taken literally.
    """
    quote = str(quote or "").strip().strip("“”\"'")
    if not quote:
        return None
    haystack = _bare(text)
    if exact:
        return _bare(quote) in haystack
    position = 0
    for piece in re.split(r"\.{3,}|…", quote):
        piece = _bare(piece)
        if not piece:
            continue
        position = haystack.find(piece, position)
        if position < 0:
            return False
        position += len(piece)
    return True


def check(fields: dict, text: str) -> dict:
    """{"quotes": n, "found": n, "missing": [field names], "shortened": n} for the fields of one answer.
    `shortened` counts the quotes that are found only because their dots are read as an abbreviation."""
    missing, quotes, shortened = [], 0, 0
    for name, field in (fields or {}).items():
        quote = (field or {}).get("evidence")
        found = in_text(quote, text)
        if found is None:
            continue
        quotes += 1
        if not found:
            missing.append(name)
        elif not in_text(quote, text, exact=True):
            shortened += 1
    return {"quotes": quotes, "found": quotes - len(missing), "missing": missing, "shortened": shortened}
