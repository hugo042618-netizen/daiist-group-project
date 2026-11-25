"""
The evidence check is for display and for the evaluation: it must not change a decision.
Out-of-scope answers show no confidence.

    uv run python tests/test_evidence.py          (or: uv run --with pytest pytest tests)
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("LOG_DIR", tempfile.mkdtemp(prefix="bidsense-test-logs-"))

import monitor  # noqa: E402
from bidsense import evidence  # noqa: E402

TEXT = "Título: Suministro y montaje de cable\nValor estimado sin IVA:  855 000,00 EUR\nTipo de procedimiento: abierto"


def test_quote_in_text_ignores_case_and_spaces():
    assert evidence.in_text("valor ESTIMADO sin iva: 855 000,00 eur", TEXT) is True
    assert evidence.in_text("Valor estimado\nsin IVA:855 000,00 EUR", TEXT) is True
    assert evidence.in_text("“Tipo de procedimiento: abierto”", TEXT) is True


def test_quote_not_in_text():
    assert evidence.in_text("equipment de cable", TEXT) is False        # a word the text does not have
    assert evidence.in_text("900 000,00 EUR", TEXT) is False
    assert evidence.in_text("", TEXT) is None and evidence.in_text(None, TEXT) is None


def test_shortened_quote():
    assert evidence.in_text("Título: Suministro...", TEXT) is True and evidence.in_text("Título: Suministro...", TEXT, exact=True) is False
    assert evidence.in_text("Suministro… Tipo de procedimiento", TEXT) is True
    assert evidence.in_text("abierto... Suministro", TEXT) is False        # the pieces must come in the text's order


def test_check_counts():
    fields = {"a": {"evidence": "Suministro y montaje"}, "b": {"evidence": "no está"}, "c": {"evidence": ""}, "d": {"evidence": "Título: Sum..."}}
    assert evidence.check(fields, TEXT) == {"quotes": 3, "found": 2, "missing": ["b"], "shortened": 1}


def test_mark_is_display_only():
    """The mark is drawn from the stored answer: confidences and the decision in the record stay as they were."""
    import copy

    import app

    agent_output = {"fields": {"cpv_code": {"value": "31300000", "confidence": 0.7, "evidence": "cable eléctrico trenzado", "evidence_found": True},
                               "contract_type": {"value": "supplies", "confidence": 0.9, "evidence": "Suministro y montaje", "evidence_found": True}}}
    before = copy.deepcopy(agent_output)
    table = app._found(agent_output, TEXT)
    assert table.count("not found in text") == 1 and "70%" in table and "90%" in table
    assert agent_output == before


def test_no_confidence_for_out_of_scope():
    logs = pd.DataFrame({
        "id": ["a", "b", "c"], "timestamp": pd.date_range("2026-11-25 10:00", periods=3, freq="min", tz="UTC"),
        "input_summary": ["x"] * 3, "review": [False] * 3, "latency_s": [2.0] * 3, "cost_eur": [0.003] * 3,
        "decision": ["BID", "OUT_OF_SCOPE", "DONT_BID"], "outputs": [{}] * 3, "feedback": pd.Series([pd.NA] * 3, dtype=object),
        "f_confidence": [0.8, 0.0, 0.6],
    })
    recent = monitor.recent(logs).set_index("decision")["confidence"]
    assert recent["OUT_OF_SCOPE"] == "" and recent["BID"] == 0.8
    confidence = monitor.quality(logs)
    confidence = confidence[confidence["signal"] == "mean confidence"]["value"]
    assert len(confidence) == 2 and abs(confidence.iloc[-1] - 0.7) < 1e-9      # the 0.0 of the out-of-scope request is not averaged in


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed.")
