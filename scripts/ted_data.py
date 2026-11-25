"""
TED open data -> the Spanish training table. Used offline only, by scripts/train.py.

Source: TED CSV open data, https://data.europa.eu/data/datasets/ted-csv (CC BY 4.0).
Column names follow its codebook ("TED CSV open data - notes & codebook", v3.6):

- contract award notices (CAN): one row per notice x lot x contract award.
  NUMBER_OFFERS ("number of tenders received") is at contract-award level.
- contract notices (CN): the call for competition. DURATION and the *estimated*
  VALUE_EURO only exist here; FUTURE_CAN_ID links a CN to its award notice.

Nothing personal is kept: winner names, addresses and ids are never read. The buyer's
name (an organisation) is read only to work out its type, and is not kept.
"""

import csv
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from bidsense import features  # noqa: E402

CAN_ZIP = config.RAW_DIR / "can_2018_2023" / "can_2018_2023.zip"
CN_ZIP = config.RAW_DIR / "cn_2018_2023" / "cn_2018_2023.zip"
CPV_RAW = config.RAW_DIR / "cpv" / "cpv.gc"
CAN_ES = config.RAW_DIR / "es_can.csv.gz"
CN_ES = config.RAW_DIR / "es_cn.csv.gz"
CPV_CSV = config.DATA_DIR / "cpv_2008.csv"

CAN_COLUMNS = [
    "ID_NOTICE_CAN", "YEAR", "CANCELLED", "ISO_COUNTRY_CODE", "TYPE_OF_CONTRACT", "TAL_LOCATION_NUTS",
    "B_FRA_AGREEMENT", "B_FRA_CONTRACT", "B_DYN_PURCH_SYST", "CPV", "ID_LOT", "LOTS_NUMBER", "VALUE_EURO",
    "TOP_TYPE", "CRIT_CODE", "CRIT_PRICE_WEIGHT", "ID_AWARD", "INFO_ON_NON_AWARD", "NUMBER_OFFERS",
    "AWARD_EST_VALUE_EURO", "CAE_TYPE", "CAE_NAME",
]
CN_COLUMNS = [
    "ID_NOTICE_CN", "YEAR", "CANCELLED", "FUTURE_CAN_ID", "ISO_COUNTRY_CODE", "ID_LOT", "LOTS_NUMBER",
    "VALUE_EURO", "DURATION", "CRIT_CODE", "CRIT_PRICE_WEIGHT",
]

# TOP_TYPE -> the agent's `procedure`. Awards without a call for competition
# (NOC/NOP/AWP) are dropped: nobody can bid for those, so they are out of scope.
PROCEDURE = {"OPE": "open", "RES": "restricted", "NIC": "negotiated", "NIP": "negotiated",
             "NEG": "negotiated", "NEC": "negotiated", "COD": "other", "INP": "other"}
NO_CALL = {"NOC", "NOP", "AWP"}
CONTRACT_TYPE = {"W": "works", "U": "supplies", "S": "services"}
MAX_OFFERS = 200  # above this, NUMBER_OFFERS is a typing error in practice (999, a year, an amount...)


# --- Download -------------------------------------------------------------------

def download(url: str, path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {path.relative_to(config.ROOT)} (large file, a few minutes)...", flush=True)
    part = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, part.open("wb") as f:  # noqa: S310 - fixed https URLs from config
        while chunk := response.read(1 << 20):
            f.write(chunk)
    part.rename(path)


def extract_country(zip_path: Path, columns: list[str], out_path: Path) -> None:
    """Keep only the rows and columns we need from a TED CSV inside its zip (about 4 GB unzipped)."""
    if out_path.is_file():
        return
    print(f"Filtering {zip_path.name} to {config.COUNTRY} rows...", flush=True)
    kept, total = [], 0
    with zipfile.ZipFile(zip_path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(name) as f:
            for chunk in pd.read_csv(f, usecols=columns, dtype=str, chunksize=250_000, encoding="utf-8"):
                total += len(chunk)
                kept.append(chunk[chunk["ISO_COUNTRY_CODE"] == config.COUNTRY])
                print(f"  {total:,} rows read", end="\r", flush=True)
    table = pd.concat(kept, ignore_index=True)
    table.to_csv(out_path, index=False, compression="gzip")
    print(f"  {total:,} rows read, {len(table):,} from {config.COUNTRY} -> {out_path.relative_to(config.ROOT)}")


def build_cpv_csv() -> None:
    """The official CPV 2008 list as a small CSV: code, Spanish label, English label."""
    if CPV_CSV.is_file():
        return
    download(config.CPV_URL, CPV_RAW)
    rows = []
    for _, element in ET.iterparse(CPV_RAW):
        if element.tag == "Row":
            values = {v.get("ColumnRef"): (v.findtext("SimpleValue") or "").strip() for v in element.findall("Value")}
            rows.append((values["code"], values.get("spa_label", ""), values.get("eng_label") or values.get("Name", "")))
            element.clear()
    rows.sort()
    CPV_CSV.parent.mkdir(parents=True, exist_ok=True)
    with CPV_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "es", "en"])
        writer.writerows(rows)
    print(f"Wrote {CPV_CSV.relative_to(config.ROOT)}: {len(rows):,} CPV codes.")


def prepare_raw() -> None:
    download(config.TED_CAN_URL, CAN_ZIP)
    download(config.TED_CN_URL, CN_ZIP)
    extract_country(CAN_ZIP, CAN_COLUMNS, CAN_ES)
    extract_country(CN_ZIP, CN_COLUMNS, CN_ES)
    build_cpv_csv()


# --- The training table -----------------------------------------------------------

def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _price_weight(rows: pd.DataFrame) -> pd.Series:
    """Weight of price in the award criteria, 0-100, per row (lot level in the data).

    CRIT_PRICE_WEIGHT is free text ("40", "40 %", "0,6", "60 puntos"...). Lowest-price
    awards (CRIT_CODE = L) count as 100.
    """
    text = rows["CRIT_PRICE_WEIGHT"].fillna("").str.replace(",", ".", regex=False)
    weight = _num(text.str.extract(r"(\d+(?:\.\d+)?)")[0])
    weight = weight.where(weight > 1, weight * 100)          # "0.6" means 60%
    weight = weight.where((weight > 0) & (weight <= 100))
    return weight.fillna(rows["CRIT_CODE"].map({"L": 100.0}))


def _region(nuts: pd.Series) -> pd.Series:
    """First NUTS code of the place of performance, cut to NUTS-2 (the Spanish region)."""
    first = nuts.fillna("").str.split(r"[^A-Z0-9]+", regex=True).str[0].str[:4]
    return first.where(first.isin(list(features.REGIONS)))


_CITY = r"ayuntamiento|ajuntament|concello|\budala?\b|alcald[ií]a|junta de gobierno local|[áa]rea de gobierno"
# Local entities that are not town halls: TED's code list "buyer legal type" files them as local authorities too.
# ("Diputación General de Aragón" is the regional government of Aragon, so it is left out.)
_LOCAL = (r"diputaci[oó]n?(?! general)\b|cabildo|consell insular|consejo insular|mancomuni(?:dad|tat)|"
          r"\bcomarca\b|conse(?:ll|jo) comarcal|foru aldundi")
_REGIONAL = (r"generalitat|xunta|junta de (?:andaluc|castilla|extremadura|comunidades)|gobierno (?:de|del|vasco)|govern de|eusko jaurlaritza|"
             r"comunidad (?:de madrid|aut[oó]noma|foral)|consejer[ií]a|conseller[ií]a|\bdepartament|principado de asturias|"
             r"servicio \w+ de (?:salud|salut)|servei \w+ de salut|servicio de salud|osakidetza|sergas|gerencia regional")


def buyer_type(cae_type: pd.Series, cae_name: pd.Series) -> pd.Series:
    """city_council / regional_government / ministry / other: the classes of the agent's prompt.

    The classes follow TED's code list "buyer legal type" (central government, regional and local
    authority, everything else). The TED CSV of 2018-2023 has an older code (CAE_TYPE) that puts
    regional and local authorities in one class and is filled in loosely, so the buyer's name decides
    first and the code only breaks ties: national authorities (1, N) are "ministry", regional or local
    ones (3, R) "regional_government", and by name town halls and the other local entities
    (provincial and island councils, comarcas, mancomunidades) are "city_council".
    """
    name = cae_name.fillna("").str.split("---").str[0].str.lower()
    local = name.str.contains(_CITY, regex=True) | name.str.contains(_LOCAL, regex=True)
    regional = name.str.contains(_REGIONAL, regex=True)
    national = cae_type.isin(["1", "N"]) | name.str.contains(r"ministerio", regex=True)
    result = pd.Series("other", index=cae_type.index)
    result[cae_type.isin(["3", "R"])] = "regional_government"
    result[national] = "ministry"
    result[regional] = "regional_government"
    result[local] = "city_council"
    return result


def build_table() -> tuple[pd.DataFrame, dict]:
    """One row per Spanish award notice, with agent-style fields and the target `offers`.

    Returns the table and the counts behind it (for the checkpoint and REPORT.md).
    """
    can = pd.read_csv(CAN_ES, dtype=str)
    counts = {"can_rows": len(can), "can_notices": can["ID_NOTICE_CAN"].nunique()}

    can = can[can["CANCELLED"] != "1"]
    can = can[~can["TOP_TYPE"].isin(NO_CALL)]
    counts["rows_with_call_for_competition"] = len(can)
    counts["notices_with_call_for_competition"] = can["ID_NOTICE_CAN"].nunique()

    # Contract awards: one row each, awarded, with a valid number of offers.
    awards = can.drop_duplicates(["ID_NOTICE_CAN", "ID_AWARD"])
    awards = awards[awards["INFO_ON_NON_AWARD"].isna()]
    counts["awards"] = len(awards)
    awards = awards.assign(offers=_num(awards["NUMBER_OFFERS"]), award_est=_num(awards["AWARD_EST_VALUE_EURO"]))
    valid = awards[(awards["offers"] >= 1) & (awards["offers"] <= MAX_OFFERS)]
    counts["awards_valid_offers"] = len(valid)
    counts["share_awards_valid_offers"] = round(len(valid) / len(awards), 4)

    target = valid.groupby("ID_NOTICE_CAN").agg(offers=("offers", "median"), n_awards=("offers", "size"))
    est = awards.groupby("ID_NOTICE_CAN")["award_est"].sum(min_count=1).rename("award_est_sum")

    lots = can.drop_duplicates(["ID_NOTICE_CAN", "ID_LOT"])
    can_weight = lots.assign(w=_price_weight(lots)).groupby("ID_NOTICE_CAN")["w"].median().rename("can_price_weight")

    notice = can.drop_duplicates("ID_NOTICE_CAN").set_index("ID_NOTICE_CAN")
    counts["notices_valid_offers"] = len(target)
    counts["share_notices_valid_offers"] = round(len(target) / len(notice), 4)
    table = notice.join(target, how="inner").join(est).join(can_weight)

    # The call for competition that led to each award: estimated value, duration, lots.
    cn = pd.read_csv(CN_ES, dtype=str)
    cn = cn[(cn["CANCELLED"] != "1") & cn["FUTURE_CAN_ID"].notna()]
    cn = cn.assign(value=_num(cn["VALUE_EURO"]), duration=_num(cn["DURATION"]),
                   lots=_num(cn["LOTS_NUMBER"]), w=_price_weight(cn))
    cn_notice = cn.groupby("FUTURE_CAN_ID").agg(
        cn_value=("value", "min"), cn_duration=("duration", "median"),
        cn_lots=("lots", "max"), cn_price_weight=("w", "median"),
    )
    table = table.join(cn_notice)
    counts["notices_linked_to_a_contract_notice"] = int(table["cn_lots"].notna().sum())

    out = pd.DataFrame(index=table.index)
    out["year"] = _num(table["YEAR"]).astype(int)
    out["offers"] = table["offers"]
    out["cpv_code"] = table["CPV"].fillna("").str.zfill(8)
    out["contract_type"] = table["TYPE_OF_CONTRACT"].map(CONTRACT_TYPE)
    out["procedure"] = table["TOP_TYPE"].map(PROCEDURE).fillna("other")
    out["region"] = _region(table["TAL_LOCATION_NUTS"])
    # Estimated value as a bidder sees it before the award: the contract notice's value,
    # else the estimate declared in the award notice. Never the final awarded price.
    out["estimated_value_eur"] = table["cn_value"].where(table["cn_value"] >= features.MIN_VALUE_EUR, table["award_est_sum"])
    out["final_value_eur"] = _num(table["VALUE_EURO"])      # kept for analysis only, not a feature
    out["duration_months"] = table["cn_duration"]
    out["lots_number"] = table["cn_lots"].where(table["cn_lots"] >= 1, _num(table["LOTS_NUMBER"])).clip(lower=1)
    out["price_weight_pct"] = table["cn_price_weight"].fillna(table["can_price_weight"])
    out["is_framework_agreement"] = (table["B_FRA_AGREEMENT"] == "Y")
    out["buyer_type"] = buyer_type(table["CAE_TYPE"], table["CAE_NAME"])
    out = out[out["cpv_code"].map(features.cpv_division).notna()]
    counts["usable_notices"] = len(out)
    counts["by_year"] = {int(y): int(n) for y, n in out["year"].value_counts().sort_index().items()}
    return out.reset_index(names="notice_id"), counts


if __name__ == "__main__":
    prepare_raw()
    table, counts = build_table()
    print(pd.Series({k: v for k, v in counts.items() if k != "by_year"}).to_string())
    print("by year:", counts["by_year"])
    print(table.describe(include="all").T.to_string())
    print(np.round(features.matrix(table).isna().mean(), 3).to_string())
