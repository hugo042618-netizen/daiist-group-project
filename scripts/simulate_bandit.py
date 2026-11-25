"""
Bonus: a SIMULATED warm-up for the prompt bandit. `uv run python scripts/simulate_bandit.py --p-a 0.70 --p-b 0.75`

The app gets little traffic, so the policy may start from simulated feedback. The
simulation is deliberately simple and kept apart from real feedback:

- each prompt gets a thumbs-up probability that the TEAM chooses and justifies (for
  example from each prompt's field accuracy in the evaluation): --p-a and --p-b;
- the epsilon-greedy policy of bidsense/bandit.py plays --rounds simulated requests;
- the resulting counts go to monitoring/bandit_warmup.json, marked as simulated.

The Monitoring tab shows simulated and real counts in separate columns. Delete the
file to start from real feedback only. Nothing here calls Gemini.
"""

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from bidsense import bandit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p-a", type=float, required=True, help="assumed thumbs-up probability of the default prompt (TEAM TO SET)")
    parser.add_argument("--p-b", type=float, required=True, help="assumed thumbs-up probability of the step-by-step prompt (TEAM TO SET)")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    probability = dict(zip(bandit.ARMS, (args.p_a, args.p_b)))
    counts = {arm: {"n": 0, "reward": 0.0} for arm in bandit.ARMS}
    for _ in range(args.rounds):
        untried = [arm for arm in bandit.ARMS if counts[arm]["n"] == 0]
        if untried:
            arm = rng.choice(untried)
        elif rng.random() < config.BANDIT_EPSILON:
            arm = rng.choice(bandit.ARMS)
        else:
            arm = max(bandit.ARMS, key=lambda a: counts[a]["reward"] / counts[a]["n"])
        counts[arm]["n"] += 1
        counts[arm]["reward"] += 1.0 if rng.random() < probability[arm] else 0.0

    bandit.WARMUP_FILE.parent.mkdir(exist_ok=True)
    bandit.WARMUP_FILE.write_text(json.dumps({
        "simulated": True, "created": date.today().isoformat(), "rounds": args.rounds, "epsilon": config.BANDIT_EPSILON,
        "assumed_thumbs_up_probability": probability, "seed": args.seed, "arms": counts,
    }, indent=1), encoding="utf-8")
    for arm, row in counts.items():
        print(f"{arm}: {row['n']} simulated pulls, mean reward {row['reward'] / max(row['n'], 1):.3f}")
    print(f"Wrote {bandit.WARMUP_FILE.relative_to(config.ROOT)} (simulated, kept apart from real feedback).")


if __name__ == "__main__":
    main()
