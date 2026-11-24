"""
Fill the Gemini cache for the evaluation inputs. `uv run python scripts/warm_cache.py`

Runs the agent on every input of eval/reference.csv, eval/shifted.csv, eval/cases.csv and
the first notices of eval/extraction_truth.csv, and stores each answer in cache/ (commit it).
After this, `main.py reference`, `drift-demo`, `evaluate` and `check.py` need no Gemini call
for those inputs, so they are fast and do not depend on the free-tier quota.

It matters for `main.py reference` in particular: if the agent fails there, the features of
that input are silently empty. This script retries and exits with an error if any input
could not be read, so run it first, and run it again after editing the eval/ files.
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import env  # noqa: E402
import evaluate  # noqa: E402
import pipeline  # noqa: E402
from bidsense import agent  # noqa: E402
from bidsense.redact import redact  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ATTEMPTS = 4


def inputs() -> dict[str, list[str]]:
    groups = {}
    for name, limit in (("reference.csv", None), ("shifted.csv", None), ("cases.csv", None), ("extraction_truth.csv", evaluate.EXTRACTION_SAMPLE)):
        path = ROOT / "eval" / name
        with path.open(newline="", encoding="utf-8") as f:
            rows = [r["input"] for r in csv.DictReader(f) if (r.get("input") or "").strip()]
        groups[name] = rows[:limit]
    return groups


def main() -> None:
    env.load_env()
    failed = 0
    for name, texts in inputs().items():
        done = cached = 0
        for i, raw in enumerate(texts, 1):
            text = redact(pipeline.read_input(raw))[0]
            for attempt in range(ATTEMPTS):
                try:
                    result = agent.extract(text)
                    done += 1
                    cached += result["cached"]
                    break
                except agent.AgentError as error:
                    if attempt == ATTEMPTS - 1:
                        failed += 1
                        print(f"\n  {name} input {i}: {str(error)[:160]}", flush=True)
                    else:
                        time.sleep(45)   # Gemini's free tier is often busy: wait and try again
            print(f"  {name}: {i}/{len(texts)}", end="\r", flush=True)
        print(f"{name}: {done}/{len(texts)} read ({cached} already cached).      ", flush=True)
    if failed:
        raise SystemExit(f"{failed} inputs could not be read. Run the script again (the cache keeps what is done).")
    print("All inputs are cached. Commit cache/.")


if __name__ == "__main__":
    main()
