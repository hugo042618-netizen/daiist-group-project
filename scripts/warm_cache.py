"""
Fill the Gemini cache for the evaluation inputs. `uv run python scripts/warm_cache.py`

Runs the agent on every input of eval/reference.csv, eval/shifted.csv, eval/cases.csv and
the first notices of eval/extraction_truth.csv, and stores each answer in cache/ (commit it).
After this, `main.py reference`, `drift-demo`, `evaluate` and `check.py` need no Gemini call
for those inputs, so they are fast and do not depend on the free-tier quota.

It matters for `main.py reference` in particular: if the agent fails there, the features of
that input are silently empty. This script retries, and exits with an error if any input
could not be read: run it first, and run it again after editing the eval/ files.

- Inputs already in the cache are skipped, so it can be stopped and started again at any time.
- The test cases and the extraction notices are read by the app's default model only, because
  they measure it; the reference and shifted inputs may be read by a fallback model.
- Requests are throttled below the free tier's per-minute limit (config.FREE_TIER_LIMITS).
- If every model has used up its free requests for the day, it stops cleanly (exit code 2) and
  says when the quotas reset; with --resume-after-reset it waits until then and carries on.
- cache/agent_models.csv records which model answered each input.
"""

import argparse
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import env  # noqa: E402
import evaluate  # noqa: E402
import pipeline  # noqa: E402
from bidsense import agent, cache  # noqa: E402
from bidsense.redact import redact  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX = config.CACHE_DIR / "agent_models.csv"
ATTEMPTS = 4
DEFAULT_MODEL_ONLY = {"cases.csv", "extraction_truth.csv"}   # files that measure the default model
MADRID = ZoneInfo("Europe/Madrid")
quota_reached = threading.Event()


def inputs() -> dict[str, list[str]]:
    groups = {}
    for name, limit in (("reference.csv", None), ("shifted.csv", None), ("cases.csv", None), ("extraction_truth.csv", evaluate.EXTRACTION_SAMPLE)):
        path = ROOT / "eval" / name
        with path.open(newline="", encoding="utf-8") as f:
            rows = [r["input"] for r in csv.DictReader(f) if (r.get("input") or "").strip()]
        groups[name] = rows[:limit]
    return groups


def cached_answer(raw: str, default_model_only: bool = False) -> dict | None:
    text = redact(pipeline.read_input(raw))[0][: config.MAX_INPUT_CHARS]
    answer = cache.get("agent", cache.key(config.GEMINI_MODEL, config.GEMINI_THINKING, config.PROMPT_VERSION, text))
    if answer and default_model_only and answer.get("model") != config.GEMINI_MODEL:
        return None
    return answer


def read_one(job: tuple[str, bool]) -> str:
    """'cached', 'read', 'quota' (daily quota used up) or an error message, for one input."""
    raw, default_model_only = job
    if cached_answer(raw, default_model_only):
        return "cached"
    text = redact(pipeline.read_input(raw))[0]
    for attempt in range(ATTEMPTS):
        if quota_reached.is_set():
            return "quota"
        try:
            agent.extract(text, default_model_only=default_model_only)
            return "read"
        except agent.DailyQuotaExhausted:
            quota_reached.set()
            return "quota"
        except agent.AgentError as error:
            if attempt == ATTEMPTS - 1:
                return str(error)[:160]
            time.sleep(45)   # Gemini's free tier is often busy: wait and try again
    return "failed"


def write_index(groups: dict[str, list[str]]) -> dict[str, int]:
    """cache/agent_models.csv: which model answered each input. Returns the number of inputs per model."""
    per_model, rows = {}, []
    for name, texts in groups.items():
        for i, raw in enumerate(texts, 1):
            answer = cached_answer(raw) or {}
            model = answer.get("model", "")
            if model:
                per_model[model] = per_model.get(model, 0) + 1
            rows.append({"file": name, "input": i, "model": model or "not cached", "requests": answer.get("requests", ""),
                         "tokens_in": answer.get("tokens_in", ""), "tokens_out": answer.get("tokens_out", ""),
                         "latency_s": answer.get("latency_s", "")})
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with INDEX.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)
    return per_model


def one_pass(groups: dict[str, list[str]], workers: int) -> tuple[int, int]:
    """Read everything that is not cached yet. Returns (inputs that failed, inputs left because of the daily quota)."""
    failed = left = 0
    for name, texts in groups.items():
        counts = {"cached": 0, "read": 0, "quota": 0}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            quota_reached.clear()   # one file being out of quota (default model only) does not stop the others
            for i, outcome in enumerate(pool.map(read_one, [(raw, name in DEFAULT_MODEL_ONLY) for raw in texts]), 1):
                if outcome in counts:
                    counts[outcome] += 1
                else:
                    failed += 1
                    print(f"\n  {name} input {i}: {outcome}", flush=True)
                print(f"  {name}: {i}/{len(texts)}", end="\r", flush=True)
        left += counts["quota"]
        print(f"{name}: {counts['cached'] + counts['read']}/{len(texts)} in the cache ({counts['read']} read now"
              + (f", {counts['quota']} left for after the quota reset" if counts["quota"] else "") + ").      ", flush=True)
    return failed, left


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=3, help="inputs read at the same time (the per-minute throttle still applies)")
    parser.add_argument("--resume-after-reset", action="store_true",
                        help="when the daily quota is used up, wait for the reset (09:00 Madrid time) and carry on")
    args = parser.parse_args()
    env.load_env()
    groups = inputs()
    while True:
        failed, left = one_pass(groups, args.workers)
        per_model = write_index(groups)
        print("Answered by: " + (", ".join(f"{model} {n}" for model, n in sorted(per_model.items())) or "nothing yet")
              + f"  ->  {INDEX.relative_to(ROOT)}", flush=True)
        if not left:
            break
        reset = agent.next_quota_reset().astimezone(MADRID)
        print(f"\nStopped cleanly: the free requests for today are used up, {left} inputs are left. "
              f"The quotas reset at {reset:%H:%M} Madrid time on {reset:%d %b}.", flush=True)
        if not args.resume_after_reset:
            print("Run this script again after that: it skips what is already cached.")
            raise SystemExit(2)
        resume = reset + timedelta(minutes=5)
        print(f"Waiting until {resume:%H:%M} Madrid time to carry on (leave this running)...", flush=True)
        while datetime.now(MADRID) < resume:
            time.sleep(60)   # short sleeps, so the wait also ends on time after the computer wakes up
        agent._daily_out.clear()
    if failed:
        raise SystemExit(f"{failed} inputs could not be read. Run the script again (the cache keeps what is done).")
    print("All inputs are cached. Commit cache/.")


if __name__ == "__main__":
    main()
