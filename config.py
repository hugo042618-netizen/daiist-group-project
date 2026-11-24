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

MODEL_VERSION = "bidders-hgb-v1"

# TED CSV open data (https://data.europa.eu/data/datasets/ted-csv), used offline only.
TED_CAN_URL = "https://data.europa.eu/api/hub/store/data/ted-contract-award-notices-2018-2023.zip"
TED_CN_URL = "https://data.europa.eu/api/hub/store/data/ted-contract-notices-2018-2023.zip"
# Official CPV 2008 list, as published by the Publications Office in the eForms SDK.
CPV_URL = "https://raw.githubusercontent.com/OP-TED/eForms-SDK/develop/codelists/cpv.gc"
COUNTRY = "ES"

# Time split: train on the older years, validate on the next, test on the latest.
TRAIN_YEARS = (2018, 2019, 2020, 2021)
VALID_YEAR = 2022
TEST_YEAR = 2023

# The 80% range shown in the app is [q10, q90].
QUANTILE_LOW, QUANTILE_HIGH = 0.1, 0.9

# Bidder buckets used in the evaluation: 1-2, 3-5, 6+.
BIDDER_BUCKETS = ((1, 2, "1-2"), (3, 5, "3-5"), (6, None, "6+"))
