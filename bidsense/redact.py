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
# DNI / NIE: the letter is attached to the digits (or after a hyphen), so "45233120 Trabajos" (a CPV code) is not one.
NATIONAL_ID = re.compile(r"\b(?:\d{8}-?[A-HJ-NP-TV-Z]|[XYZ]-?\d{7}-?[A-HJ-NP-TV-Z])\b", re.IGNORECASE)

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


# Stricter filter, for short texts written by the model (evidence quotes, notes): a common given name
# followed by one to three capitalised words is treated as a person. It also hides a few things named
# after people ("Hospital Ramón y Cajal"), which is acceptable in a quote but not in the tender itself.
GIVEN_NAMES = (
    "maría maria josé jose juan antonio manuel francisco david javier daniel carlos jesús jesus alejandro miguel rafael pedro "
    "pablo ángel angel sergio fernando luis jorge alberto álvaro alvaro adrián adrian diego raúl raul enrique ramón ramon vicente "
    "andrés andres ignacio joaquín joaquin eduardo víctor victor roberto jaime rubén ruben óscar oscar iván ivan marcos "
    "gonzalo ricardo emilio tomás tomas agustín agustin félix felix alfonso guillermo jordi josep joan xavier marc "
    "carmen ana isabel dolores teresa cristina marta laura elena lucía lucia paula sara raquel beatriz patricia silvia "
    "rocío rocio julia irene nuria núria inmaculada susana yolanda mónica monica sonia alicia esther eva "
    "concepción concepcion encarnación encarnacion amparo natalia sandra verónica veronica margarita begoña "
    "andrea claudia noelia miriam lorena carolina olga gloria josefa francisca antonia juana manuela catalina"
).split()
GIVEN_NAME_SEQUENCE = re.compile(
    # Not after "San" / "Santa" / "Sant": those are towns (San Fernando de Henares), not people.
    rf"(?<!\bSan )(?<!\bSanta )(?<!\bSant )(?<!\bSAN )(?<!\bSANTA )"
    rf"\b(?i:{'|'.join(sorted(set(GIVEN_NAMES), key=len, reverse=True))})(?:\s+(?:de|del|de la|de los|y|i))?(?:\s+{_NAME}){{1,3}}"
)


def redact(text: str, strict: bool = False) -> tuple[str, dict[str, int]]:
    """The text with personal data replaced by [EMAIL], [PHONE], [ID], [ACCOUNT], [NAME], and what was removed.

    `strict` adds the given-name filter above: use it on what the model writes, not on the tender.
    """
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
    if strict:
        text = replace(GIVEN_NAME_SEQUENCE, "[NAME]", "name", text)
    return text, counts


def scrub(value):
    """Strict redaction of every string inside a JSON-like value (dicts and lists are walked)."""
    if isinstance(value, str):
        return redact(value, strict=True)[0]
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    return value
