"""
Which language is a text written in? A small stop-word count, no model and no API.

Used for the drift check (share of texts outside the languages of Spanish tenders) so that
it works even when the agent is down. The agent also reports the language it read.

Spanish tenders are published in Spanish and in the co-official languages, so Catalan,
Galician and Basque are in scope (config.IN_SCOPE_LANGUAGES); Portuguese, French or English
texts are not. Spanish, Catalan, Galician and Portuguese share many short words ("de", "a",
"que"), so a word that belongs to one language only counts three times.
"""

import re

STOPWORDS = {
    "es": "el la los las de del que y en un una para por con se su al lo como más o pero sus le ya este sí porque esta entre "
          "cuando muy sin sobre también hasta hay donde quien desde todo nos durante uno ni contra ese eso ante ellos esto antes "
          "algunos qué unos otro otras será según mediante no es a e son ha servicio servicios suministro contratación obras "
          "contrato años meses mantenimiento ayuntamiento valor duración procedimiento abierto acuerdo lugar ejecución criterios "
          "adjudicación precio calidad título órgano descripción lotes".split(),
    "ca": "el la els les de del dels que i en un una per amb no es al als més com però aquest aquesta seu seva són ha hi també "
          "fins des tot ser servei serveis subministrament contracte contractació mitjançant pel pels any anys manteniment obres "
          "ajuntament dels d'un d'una l'ajuntament segons durant entre sense sobre aquests aquestes seus seves".split(),
    "gl": "o a os as de do da dos das que e en un unha para por con non se ao aos á ás máis como pero este esta seu súa son hai "
          "tamén ata desde todo servizo servizos subministración contratación mediante polo pola dun dunha nos nas no na obras "
          "mantemento anos concello xunta xestión segundo durante entre sen sobre estes estas seus súas".split(),
    "eu": "eta da du dira dute bat ere edo ez bere izango egiteko zerbitzua zerbitzuak zerbitzuen kontratua kontratuaren hornidura "
          "lanak udala udalaren esleipena lizitazioa mantentze arabera bidez urte urteko hilabete duen dituen behar beharreko "
          "honen hau hori batean artean gabe izan dela diren".split(),
    "pt": "o a os as de do da dos das que e em um uma para por com não se na no ao à mais como mas foi pelo pela até isso ela "
          "entre era depois sem mesmo aos ter seus quem nas nos já está também só são você tem será serviço serviços aquisição "
          "fornecimento prestação concurso contratação obras anos manutenção município câmara segundo durante sobre".split(),
    "en": "the of and to in a is that for it as was with be by on not this but from or have an they which one you were all "
          "there been their has would what will if can when more no out other its into than only shall may such any these must "
          "should under services supply works contract maintenance".split(),
    "fr": "le la les de des du et en un une pour par avec que qui dans ce il au aux ne se pas sur plus est sont ou mais comme "
          "son sa ses leur cette ces être fait nous vous elle tout sous entre marché prestations travaux fourniture relatif "
          "réalisation services".split(),
    "it": "il lo la i gli le di del della dei che e in un una per con non si al più come ma questo sono suo sua anche fino tra "
          "essere servizi fornitura affidamento gara lavori".split(),
    "de": "der die das und in den von zu mit sich des auf für ist im dem nicht ein eine als auch es an werden aus er hat dass "
          "sie nach wird bei einer um am sind noch wie einem über".split(),
}
_SETS = {language: set(words) for language, words in STOPWORDS.items()}
_SHARED = {word for words in _SETS.values() for word in words if sum(word in other for other in _SETS.values()) > 1}
_WORD = re.compile(r"[a-zà-ÿ']+", re.IGNORECASE)
UNIQUE_WEIGHT = 3


def detect(text: str) -> str:
    """ISO code of the most likely language (es, ca, gl, eu, pt, en, fr, it, de) or "unknown"."""
    words = [w.lower() for w in _WORD.findall(str(text))][:2000]
    if len(words) < 3:
        return "unknown"
    scores = {language: sum((1 if w in _SHARED else UNIQUE_WEIGHT) for w in words if w in vocabulary)
              for language, vocabulary in _SETS.items()}
    best = max(scores, key=scores.get)   # ties go to the first language listed, Spanish
    return best if scores[best] >= max(2, 0.04 * len(words)) else "unknown"
