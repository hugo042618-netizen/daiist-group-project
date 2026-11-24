"""
Bonus (reinforcement learning): an epsilon-greedy bandit that chooses between the two
extraction prompts using the thumbs up/down the app collects.

    arm    = a prompt version (bidsense/prompts.py)
    reward = 1 for thumbs up, 0 for thumbs down, on the request that used that prompt
    policy = with probability epsilon pick a random arm, otherwise the arm with the best
             average reward so far

The counts are rebuilt from the request and feedback logs each time, so the policy
survives restarts and learns only from what is in the logs. A simulated warm-up
(scripts/simulate_bandit.py) is kept in its own file, monitoring/bandit_warmup.json, and
is reported separately from real feedback. Off by default: config.BANDIT_ENABLED.
"""

import json
import random

import config
from bidsense import prompts

WARMUP_FILE = config.ROOT / "monitoring" / "bandit_warmup.json"
ARMS = list(prompts.PROMPTS)


def _empty() -> dict:
    return {arm: {"n": 0, "reward": 0.0} for arm in ARMS}


def warmup_counts() -> dict:
    """Simulated pulls and rewards per arm (not real users)."""
    counts = _empty()
    if WARMUP_FILE.is_file():
        try:
            for arm, row in json.loads(WARMUP_FILE.read_text(encoding="utf-8")).get("arms", {}).items():
                if arm in counts:
                    counts[arm] = {"n": int(row["n"]), "reward": float(row["reward"])}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    return counts


def real_counts(logs=None) -> dict:
    """Rated requests and thumbs up per arm, from the app's own logs."""
    counts = _empty()
    if logs is None:
        import logbook

        logs = logbook.load()
    if logs is None or logs.empty or "versions" not in logs or "feedback" not in logs:
        return counts
    rated = logs.dropna(subset=["feedback"])
    for versions, feedback in zip(rated["versions"], rated["feedback"]):
        arm = (versions or {}).get("prompt_version") if isinstance(versions, dict) else None
        if arm in counts:
            counts[arm]["n"] += 1
            counts[arm]["reward"] += 1.0 if float(feedback) > 0 else 0.0
    return counts


def state(logs=None) -> dict:
    """Per arm: simulated and real counts, and the average reward the policy uses."""
    warmup, real = warmup_counts(), real_counts(logs)
    table = {}
    for arm in ARMS:
        n = warmup[arm]["n"] + real[arm]["n"]
        reward = warmup[arm]["reward"] + real[arm]["reward"]
        table[arm] = {"simulated_n": warmup[arm]["n"], "real_n": real[arm]["n"],
                      "real_thumbs_up": real[arm]["reward"], "mean_reward": round(reward / n, 3) if n else None}
    return table


def choose(logs=None, rng: random.Random | None = None) -> str:
    """The prompt version for the next request."""
    if not config.BANDIT_ENABLED:
        return config.PROMPT_VERSION
    rng = rng or random
    try:
        table = state(logs)
    except Exception:  # noqa: BLE001 - the logs being unreachable must not break a request
        return config.PROMPT_VERSION
    untried = [arm for arm in ARMS if table[arm]["mean_reward"] is None]
    if untried:
        return rng.choice(untried)
    if rng.random() < config.BANDIT_EPSILON:
        return rng.choice(ARMS)
    return max(ARMS, key=lambda arm: table[arm]["mean_reward"])
