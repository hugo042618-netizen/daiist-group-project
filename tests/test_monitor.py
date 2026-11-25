"""
The Monitoring tab must never show an error because a value is missing in the logs.

    uv run python tests/test_monitor.py          (or: uv run --with pytest pytest tests)

The bug this guards against (25 Nov 2026, on Render): with requests logged and no thumbs yet,
`logbook.load()` returns a `feedback` column of pd.NA, and `monitor.quality()` did
`.astype(float)` on it: TypeError, and every panel of the tab showed "Error".
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("LOG_DIR", tempfile.mkdtemp(prefix="bidsense-test-logs-"))   # never the real logs or the database

import monitor  # noqa: E402


def logs(feedback: list, review=None) -> pd.DataFrame:
    """Logged requests as logbook.load() returns them, one per feedback value."""
    n = len(feedback)
    frame = pd.DataFrame({
        "id": [f"r{i}" for i in range(n)],
        "timestamp": pd.date_range("2026-11-25 10:00", periods=n, freq="min", tz="UTC"),
        "input_summary": ["cpv 45 | €100k-500k | es | 600 chars"] * n,
        "review": review if review is not None else [i % 3 == 0 for i in range(n)],
        "latency_s": [3.2] * n,
        "cost_eur": [0.0036] * n,
        "decision": ["BID"] * n,
        "outputs": [{"model": {"expected_bidders": 2.4}, "decision": {"decision": "BID"}}] * n,
        "versions": [{"prompt_version": "extract-v3"}] * n,
        "f_language_in_scope": [1.0] * n, "f_cpv_division": [45.0] * n, "f_log_value": [5.6] * n,
        "f_text_length": [600.0] * n, "f_confidence": [0.8] * n,
    })
    frame["feedback"] = pd.Series(feedback, dtype=object)
    return frame


REFERENCE = pd.DataFrame({"language_in_scope": [1.0] * 40, "cpv_division": list(np.tile([45.0, 72.0, 33.0, 79.0], 10)),
                          "log_value": list(np.linspace(5.0, 6.3, 40)), "text_length": list(np.linspace(300, 900, 40)),
                          "confidence": [0.8] * 40})


def every_panel(frame: pd.DataFrame) -> dict:
    return {"kpis": monitor.kpis(frame), "quality": monitor.quality(frame), "summary": monitor.summary(frame),
            "recent": monitor.recent(frame), "alerts": monitor.alerts(frame, REFERENCE),
            "drift": monitor.drift(REFERENCE, monitor.live_features(frame)) if not frame.empty else None}


def test_no_feedback_at_all():
    """What Postgres gave on Render: requests, and pd.NA in every feedback cell."""
    panels = every_panel(logs([pd.NA] * 25))
    assert panels["kpis"]["requests"] == 25 and panels["kpis"]["rated"] == 0 and panels["kpis"]["thumbs_down_share"] is None
    signals = set(panels["quality"]["signal"])
    assert "share sent to review" in signals and "mean confidence" in signals
    assert not any("👎" in signal for signal in signals)          # nothing rated, so no thumbs-down line
    assert list(panels["recent"]["feedback"]) == [""] * 25
    assert not any("👎" in message for message in panels["alerts"])   # drift may alert on these made-up rows; feedback cannot


def test_one_missing_feedback_among_rated():
    panels = every_panel(logs([1, pd.NA, -1, None, float("nan"), 1]))
    assert panels["kpis"]["rated"] == 3
    assert abs(panels["kpis"]["thumbs_down_share"] - 1 / 3) < 1e-9
    thumbs = panels["quality"][panels["quality"]["signal"].str.contains("👎")]
    assert len(thumbs) == 3 and thumbs["value"].between(0, 1).all()
    assert sorted(panels["recent"]["feedback"]) == ["", "", "", "👍", "👍", "👎"]


def test_missing_numbers_elsewhere():
    """None in review, cost or latency (a record written by an older version) must not break a panel either."""
    frame = logs([pd.NA, 1, pd.NA], review=[True, None, False])
    frame["cost_eur"] = pd.Series([0.004, None, pd.NA], dtype=object)
    frame["latency_s"] = pd.Series([None, None, None], dtype=object)
    frame["f_confidence"] = pd.Series([0.9, pd.NA, None], dtype=object)
    panels = every_panel(frame)
    assert panels["kpis"]["mean_latency_s"] is None and panels["kpis"]["review_share"] == 0.5
    assert abs(panels["kpis"]["mean_cost_eur"] - 0.004) < 1e-12
    assert not panels["quality"]["value"].isna().any()


def test_no_requests():
    panels = every_panel(pd.DataFrame())
    assert panels["kpis"]["requests"] == 0 and panels["quality"].empty and panels["recent"].empty and panels["alerts"] == []


def test_monitoring_tab_with_unrated_requests():
    """The whole tab, through app.on_refresh(), on logs with requests and no feedback file."""
    import app
    import logbook

    for i in range(3):
        logbook.log_request({"input_summary": "cpv 45 | €100k-500k | es | 600 chars", "decision": "BID",
                             "features": {"language_in_scope": 1.0, "cpv_division": 45.0, "log_value": 5.6, "text_length": 600.0, "confidence": 0.8},
                             "outputs": {"agent": {}, "model": {"expected_bidders": 2.4}, "decision": {"decision": "BID"}, "baseline": None},
                             "versions": {"prompt_version": "extract-v3"}, "review": i == 0, "latency_s": 3.2, "tokens": 9000, "cost_eur": 0.0036})
    loaded = logbook.load()
    assert len(loaded) >= 3 and loaded["feedback"].isna().all()
    panels = app.on_refresh()
    assert len(panels) == 8
    assert "Requests" in panels[0] and "Error" not in panels[0]
    assert "waiting for data" in panels[1]                          # fewer than the 20 requests the drift check needs
    assert panels[2]["visible"] is True and panels[3]["visible"] is False   # the chart is shown, its "No data yet" is not
    assert "No data yet" in panels[4]
    assert "<table" in panels[5] and panels[5].count("<tr>") >= 4       # the header row and the three requests


def test_monitoring_tab_empty_states():
    """With no logs at all every panel says "No data yet" instead of failing."""
    import app
    import logbook

    saved, logbook.LOG_DIR = logbook.LOG_DIR, Path(tempfile.mkdtemp(prefix="bidsense-test-empty-"))
    try:
        panels = app.on_refresh()
    finally:
        logbook.LOG_DIR = saved
    assert "Requests" in panels[0]
    assert panels[2]["visible"] is False and "No data yet" in panels[3]["value"] and panels[3]["visible"] is True
    assert "No data yet" in panels[4]
    assert "No data yet" in panels[5] and "No data yet" in panels[7]["value"]


def test_drift_banner_is_green_or_red():
    """With enough requests the banner is green when they look like the reference set and red when they do not."""
    import app

    reference = monitor.load_reference()
    assert reference is not None and len(reference) >= 20, "monitoring/reference_features.csv is committed with the app"
    normal = logs([pd.NA] * len(reference))
    for name in reference.columns:
        normal[f"f_{name}"] = reference[name].to_numpy()
    assert "bs-ok" in app._drift_banner(normal, reference) and "No drift" in app._drift_banner(normal, reference)
    foreign = normal.assign(f_language_in_scope=0.0)
    banner = app._drift_banner(foreign, reference)
    assert "bs-alert" in banner and "Alert" in banner and "language" in banner
    assert "alert</span>" in app._drift_table(foreign, reference) and "alert</span>" not in app._drift_table(normal, reference)
    assert "waiting for data" in app._drift_banner(normal.head(5), reference)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed.")
