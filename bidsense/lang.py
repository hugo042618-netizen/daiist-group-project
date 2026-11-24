"""
Which language is a text written in? A small stop-word count, no model and no API.

Used for the drift check (share of non-Spanish texts) so that it works even when the
agent is down. The agent also reports the language it read; the two can be compared.
"""

import re

STOPWORDS = {
    "es": "el la los las de del que y en un una para por con se su al lo como más o pero sus le ya este "
          "sí porque esta entre cuando muy sin sobre también hasta hay donde quien desde todo nos durante "
          "uno ni contra ese eso ante ellos esto antes algunos qué unos otro otras será según mediante".split(),
    "en": "the of and to in a is that for it as was with be by on not this but from or have an they which "
          "one you were all there been their has would what will if can when more no out other its into "
          "than only shall may such any these must should under".split(),
    "pt": "o a os as de do da dos das que e em um uma para por com não se na no ao à mais como mas foi "
          "pelo pela até isso ela entre era depois sem mesmo aos ter seus quem nas nos já está também só "
          "são você tem será serviços aquisição fornecimento prestação concurso".split(),
    "fr": "le la les de des du et en un une pour par avec que qui dans ce il au aux ne se pas sur plus "
          "est sont ou mais comme son sa ses leur cette ces être fait nous vous elle tout sous entre "
          "marché prestations travaux fourniture relatif réalisation".split(),
    "ca": "el la els les de del dels que i en un una per amb no es al als més com però aquest aquesta "
          "seu seva són ha hi també fins des tot ser serveis subministrament contracte mitjançant".split(),
    "it": "il lo la i gli le di del della dei che e in un una per con non si al più come ma questo "
          "sono suo sua anche fino tra essere servizi fornitura affidamento gara".split(),
    "de": "der die das und in den von zu mit sich des auf für ist im dem nicht ein eine als auch es an "
          "werden aus er hat dass sie nach wird bei einer um am sind noch wie einem über".split(),
}
_SETS = {language: set(words) for language, words in STOPWORDS.items()}
_WORD = re.compile(r"[a-zà-ÿ]+", re.IGNORECASE)


def detect(text: str) -> str:
    """ISO code of the most likely language (es, en, pt, fr, ca, it, de) or "unknown"."""
    words = [w.lower() for w in _WORD.findall(str(text))][:2000]
    if len(words) < 3:
        return "unknown"
    scores = {language: sum(w in vocabulary for w in words) for language, vocabulary in _SETS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= max(2, 0.04 * len(words)) else "unknown"
