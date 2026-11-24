"""
All the settings of the product in one place.

Values marked TEAM TO SET are temporary defaults: the team decides them and
justifies them in REPORT.md.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # TED downloads, git-ignored
MODELS_DIR = ROOT / "models"

# --- Component 2: the bidders model (scripts/train.py) -------------------------

MODEL_VERSION = "bidders-hgb-v2"

# TED CSV open data (https://data.europa.eu/data/datasets/ted-csv), used offline only.
TED_CAN_URL = "https://data.europa.eu/api/hub/store/data/ted-contract-award-notices-2018-2023.zip"
TED_CN_URL = "https://data.europa.eu/api/hub/store/data/ted-contract-notices-2018-2023.zip"
# Official CPV 2008 list, as published by the Publications Office in the eForms SDK.
CPV_URL = "https://raw.githubusercontent.com/OP-TED/eForms-SDK/develop/codelists/cpv.gc"
COUNTRY = "ES"

# Time split: settings are chosen by training on TRAIN_YEARS and validating on VALID_YEAR;
# the final models are then refitted on both (2018-2022) and tested once on TEST_YEAR.
TRAIN_YEARS = (2018, 2019, 2020, 2021)
VALID_YEAR = 2022
TEST_YEAR = 2023
FINAL_TRAIN_YEARS = TRAIN_YEARS + (VALID_YEAR,)

# The 80% range shown in the app is [q10, q90].
QUANTILE_LOW, QUANTILE_HIGH = 0.1, 0.9

# Bidder buckets used in the evaluation: 1-2, 3-5, 6+.
BIDDER_BUCKETS = ((1, 2, "1-2"), (3, 5, "3-5"), (6, None, "6+"))

# The app shows the MEDIAN model's prediction as "expected bidders" and uses it for the
# win chance (team decision, 24 Nov 2026: it is the model that beats the baseline on MAE).
POINT_MODEL = "median"

# --- Component 1: the Gemini agent (bidsense/agent.py) ------------------------

# The free tier allows only 20 requests a day per project on gemini-3.8-flash and gemini-3.6-flash (measured
# on 24 Nov 2026: quota "GenerateRequestsPerDayPerProjectPerModel-FreeTier"), and one tender needs about 5
# requests. gemini-3.5-flash-lite is the Flash model that the free tier can actually serve, so it is the
# default; the other two are tried only if it is unavailable. The model that answered is logged per request.
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_FALLBACK_MODELS = ("gemini-3.8-flash", "gemini-3.6-flash")
MODEL_COOLDOWN_S = 300                   # after a model is unavailable, try it last for this long (longer if its daily quota is used up)
GEMINI_TIMEOUT_S = 45                    # per call; a slow call counts as unavailable
GEMINI_THINKING = "low"                  # low / medium / high
PROMPT_VERSION = "extract-v1"            # the default extraction prompt (see bidsense/prompts.py)
MAX_INPUT_CHARS = 20_000                 # longer texts are cut before they are sent
MAX_AGENT_STEPS = 8                      # model turns per request, tool calls included
GEMINI_RETRIES = 4                       # per call: flash-lite answers in about 1 s but returns 503 now and then; waits 2, 4, 8, 16 s

# Cost estimate. The free tier costs nothing; these are Gemini's PAID-tier prices for
# GEMINI_MODEL (USD per 1M tokens, valid until 31 Dec 2026), so the logged cost is what the
# product would cost on a paid plan. Thinking tokens are billed as output.
PRICES_USD_PER_M = {                     # model: (input, output)
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}
USD_TO_EUR = 0.92                        # TEAM TO SET: exchange rate used for the estimate

# Every agent run and every judge call is cached on disk, keyed by model, prompt version
# and (redacted) text, so evaluation, `main.py reference` and `drift-demo` do not burn the
# free-tier quota. The cache of the eval inputs is committed.
CACHE_DIR = ROOT / "cache"
USE_CACHE = True

# --- Decision and review rule (bidsense/decision.py) ---------------------------
# win_chance      = min(1, 1 / expected_bidders)
# expected_value  = win_chance * contract_value * margin - bid_cost
# BID if expected_value > 0, DON'T BID if < 0, SEND TO REVIEW if any rule below applies.

DEFAULT_BID_COST_EUR = 3_000             # TEAM TO SET: cost of preparing one bid (UI default)
DEFAULT_MARGIN_PCT = 8.0                 # TEAM TO SET: profit margin on the contract value (UI default)
REVIEW_VALUE_EUR = 5_000_000             # TEAM TO SET: contracts above this always go to a person
CLOSE_CALL_EUR = 2_000                   # TEAM TO SET: |expected value| below this is too close to call
MIN_CONFIDENCE = 0.6                     # TEAM TO SET: below this the agent's reading is not trusted
KEY_FIELDS = ("cpv_code", "contract_type", "estimated_value_eur")   # TEAM TO SET: a missing one sends to review

# --- Monitoring and drift (monitor.py) -----------------------------------------

DRIFT_FEATURES = ("is_spanish", "cpv_division", "log_value", "text_length")
PSI_ALERT = 0.2                          # TEAM TO SET: alert when a feature's PSI exceeds this...
PSI_NOISE_MULTIPLE = 2                   # TEAM TO SET: ...and this many times the PSI expected by chance
NON_SPANISH_ALERT = 0.30                 # TEAM TO SET: alert when more than this share of texts is not Spanish
THUMBS_DOWN_ALERT = 0.40                 # TEAM TO SET: alert when this share of rated requests is thumbs down
DRIFT_WINDOW = 100                       # the drift check looks at the last N requests...
DRIFT_MIN_WINDOW = 20                    # ...and stays silent with fewer than this

# --- Bonus: bandit over two extraction prompts (bidsense/bandit.py) ------------

BANDIT_ENABLED = False                   # TEAM TO SET: turn on once the bonus is ready
BANDIT_EPSILON = 0.1                     # TEAM TO SET: share of requests that explore
