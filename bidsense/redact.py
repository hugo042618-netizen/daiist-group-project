"""
Remove personal data from a text BEFORE it goes to Gemini, to the logs or to the cache.

Rule-based on purpose (a name-recognition model does not fit in 512 MB): emails, phone
numbers, national ids, bank accounts, and names that follow a contact label or a
courtesy title. A person's name written in free prose, with no label, is NOT caught:
say so to users and never paste CVs or staff lists.

Organisation names (buyers, companies) are not personal data and are kept.
"""

import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
NATIONAL_ID = re.compile(r"\b(?:\d{8}[- ]?[A-HJ-NP-TV-Z]|[XYZ][- ]?\d{7}[- ]?[A-HJ-NP-TV-Z])\b", re.IGNORECASE)

_MONEY_AFTER = r"(?!\s*(?:[.,]\d|\d|EUR|€|euros?\b|millones|mil\b))"
PHONE_LABELLED = re.compile(
    r"(?i)\b((?:tel[eé]fonos?|tel[eé]f|tlfn?o?|tfno|tel|m[oó]vil|fax|phone|mobile|t[eé]l[eé]phone)\b\s*[.:]?\s*)"
    r"(\+?\d[\d ()./-]{6,18}\d)"
)
PHONE_INTERNATIONAL = re.compile(r"(?<![\w.,])(?:\+|00)\d{2,3}[ ./-]?(?:\d[ ./-]?){8,11}(?!\d)")
PHONE_SPANISH = re.compile(
    r"(?<![\d.,€])(?:[6789]\d{8}|[6789]\d{2}[ .-]\d{3}[ .-]\d{3}|[6789]\d{2}[ .-]\d{2}[ .-]\d{2}[ .-]\d{2})(?![\d])" + _MONEY_AFTER
)

_NAME = r"[A-ZÁÉÍÓÚÑÜ][\wÁÉÍÓÚÑÜáéíóúñü'’-]+"
_NAME_SEQUENCE = rf"{_NAME}(?:\s+(?:de|del|de la|de los|da|dos|do|van|von|y|i|e)?\s*{_NAME}){{0,4}}"
CONTACT_LABEL = re.compile(
    r"(?im)^(\s*(?:persona(?:s)? de contacto|punto de contacto|a la atenci[oó]n de|atenci[oó]n|contacto|responsable(?: del contrato)?|"
    r"nombre y apellidos|nombre|firmado(?: por)?|fdo\.?|contact person|contact point|contact|attn\.?|for the attention of|"
    r"pessoa de contacto|personne de contact|point de contact)\s*[:：]\s*)(.+)$"
)
COURTESY_TITLE = re.compile(
    rf"\b((?:D\.ª|D\.|Dª|Dña\.?|Don|Doña|Sr\.|Sra\.|Sres\.|Mr\.?|Mrs\.?|Ms\.?|Dr\.|Dra\.|M\.|Mme\.?)\s+){_NAME_SEQUENCE}"
)


def redact(text: str) -> tuple[str, dict[str, int]]:
    """The text with personal data replaced by [EMAIL], [PHONE], [ID], [ACCOUNT], [NAME], and what was removed."""
    counts = {}

    def replace(pattern: re.Pattern, replacement, kind: str, value: str) -> str:
        value, n = pattern.subn(replacement, value)
        if n:
            counts[kind] = counts.get(kind, 0) + n
        return value

    text = replace(EMAIL, "[EMAIL]", "email", str(text))
    text = replace(IBAN, "[ACCOUNT]", "account", text)
    text = replace(NATIONAL_ID, "[ID]", "id", text)
    text = replace(PHONE_LABELLED, r"\1[PHONE]", "phone", text)
    text = replace(PHONE_INTERNATIONAL, "[PHONE]", "phone", text)
    text = replace(PHONE_SPANISH, "[PHONE]", "phone", text)
    text = replace(CONTACT_LABEL, r"\1[NAME]", "name", text)
    text = replace(COURTESY_TITLE, r"\1[NAME]", "name", text)
    return text, counts
