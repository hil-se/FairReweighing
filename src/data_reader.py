import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_config import (
    N_SYNTHETIC,
    P_A1,
    synthetic_height_mean,
    synthetic_height_sd,
    synthetic_power_mean,
    synthetic_power_sd,
)
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores*")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def load_dataset(name, seed=None, **kwargs):
    loaders = {
        "community": load_communities_bi,
        "communities": load_communities_bi,
        "community_multi": load_communities_multi,
        "communities_multi": load_communities_multi,
        "community_con": load_communities_con,
        "community-continuous": load_communities_con,
        "insurance": load_insurance,
        "insurance_multi": load_insurance_multi,
        "lsac": load_lsac,
        "lsac_multi": load_lsac_multi,
        "german": load_german,
        "heart": load_heart,
        "synthetic": load_synthetic,
        "scut": load_scut_multi,
        "scut-fbp5500": load_scut_multi,
        "scut_sex": load_scut_sex,
        "scut-sex": load_scut_sex,
        "scut_multi": load_scut_multi,
        "scut-multi": load_scut_multi,
    }
    key = str(name).lower()
    if key not in loaders:
        raise ValueError(f"Unknown dataset: {name}")
    return loaders[key](seed=seed, **kwargs)


def load_german(seed=None, **kwargs):
    data = pd.read_csv(DATA_DIR / "german_credit_data.csv", index_col=0)
    data = data.dropna()
    data["Sex"] = data["Sex"].apply(lambda x: 1 if x == "male" else 0)
    data["Risk"] = data["Risk"].apply(lambda x: 1 if x == "good" else 0)
    dependent = "Risk"
    X = data.drop([dependent], axis=1)
    y = np.array(data[dependent])
    protected = ["Sex"]
    return X, y, protected


def load_heart(seed=None, **kwargs):
    data = pd.read_csv(DATA_DIR / "heart.csv")
    data = data.dropna()
    dependent = "output"
    X = data.drop([dependent], axis=1)
    y = np.array(data[dependent])
    protected = ["sex"]
    return X, y, protected


def load_communities_con(seed=None, **kwargs):
    column_names = ["state", "county", "community", "communityname", "fold", "population", "householdsize",
                    "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29",
                    "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf",
                    "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc", "whitePerCap",
                    "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov",
                    "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy",
                    "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce",
                    "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par",
                    "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg",
                    "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig",
                    "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell",
                    "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous",
                    "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR",
                    "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos",
                    "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal",
                    "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ", "MedRent", "MedRentPctHousInc",
                    "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn",
                    "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT",
                    "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq",
                    "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite",
                    "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits",
                    "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans", "PolicCars",
                    "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn",
                    "PolicBudgPerPop", "ViolentCrimesPerPop"]

    data = pd.read_csv(DATA_DIR / "communities.data", sep=",", names=column_names, na_values="?")

    assert not data["ViolentCrimesPerPop"].isna().any()

    labels = data["ViolentCrimesPerPop"].values.astype(np.float32)
    data = data.drop(columns=["ViolentCrimesPerPop", "state", "county", "community", "communityname", "fold"])
    data.fillna(0, inplace=True)
    features = pd.DataFrame(data.values.astype(np.float32), columns=data.columns)
    protected = ["racepctblack"]
    return features, labels, protected


def load_communities_bi(seed=None, **kwargs):
    df = pd.read_csv(DATA_DIR / "communities.csv")
    df = df.fillna(0)
    black = "racepctblack"
    white = "racePctWhite"
    asian = "racePctAsian"
    hispanic = "racePctHisp"
    df_sens = df[[black, white, asian, hispanic]]

    y = np.array(df["ViolentCrimesPerPop"])
    df = df.drop("ViolentCrimesPerPop", axis=1)

    majority = df_sens.apply(pd.Series.idxmax, axis=1)
    df["race"] = majority.map({black: 0, white: 1, asian: 0, hispanic: 0})
    df = df.drop([black, white, asian, hispanic], axis=1)
    return df, y, ["race"]


def load_communities_multi(seed=None, **kwargs):
    df = pd.read_csv(DATA_DIR / "communities.csv")
    df = df.fillna(0)
    protected = ["racepctblack", "racePctWhite", "racePctAsian", "racePctHisp"]
    y = np.array(df["ViolentCrimesPerPop"])
    X = df.drop("ViolentCrimesPerPop", axis=1)
    return X, y, protected


def load_lsac(seed=None, **kwargs):
    return _load_lsac(seed=seed, protected=["race"])


def load_lsac_multi(seed=None, **kwargs):
    return _load_lsac(seed=seed, protected=["race", "gender"])


def _load_lsac(seed=None, protected=None):
    df = pd.read_csv(DATA_DIR / "lawschool.csv", skipinitialspace=True, na_values=["NA", ""])
    df.columns = df.columns.str.strip()
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].astype(str).str.strip()
    df = df.replace({"NA": np.nan, "": np.nan, "nan": np.nan})
    model_columns = [
        "race",
        "cluster",
        "lsat",
        "ugpa",
        "zfygpa",
        "fulltime",
        "fam_inc",
        "age",
        "gender",
    ]
    for column in ["race", "cluster", "lsat", "ugpa", "zfygpa", "fulltime", "fam_inc", "age"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[model_columns].dropna().copy()
    df["race"] = (df["race"] == 7.0).astype(int)
    df["gender"] = df["gender"].map({"male": 1, "female": 0})
    df = df.dropna(subset=["gender"])

    # Keep the manuscript's 6,240-case benchmark fixed across repeated splits.
    # The earlier loader resampled the source data for every experiment seed.
    if len(df) > 6240:
        df, _ = train_test_split(
            df,
            train_size=6240,
            stratify=df["race"],
            random_state=0,
        )

    y = df.pop("zfygpa").to_numpy(dtype=float)
    return df.reset_index(drop=True), y, protected or ["race"]


def load_insurance(seed=None, **kwargs):
    return _load_insurance(protected=["age"])


def load_insurance_multi(seed=None, **kwargs):
    return _load_insurance(protected=["age", "sex"])


def _load_insurance(protected):
    data = pd.read_csv(DATA_DIR / "insurance.csv")
    data["sex"] = data["sex"].apply(lambda x: 1 if x == "male" else 0)
    dependent = "charges"
    X = data.drop(dependent, axis=1)
    X["age"] = X["age"] / max(X["age"])
    y = np.array(data[dependent])
    return X, y / max(y), protected


def load_synthetic(n=N_SYNTHETIC, p=P_A1, seed=None, **kwargs):
    rng = np.random.default_rng(seed)
    keys = ["sex", "height"]
    data = {key: [] for key in keys}
    y = []
    for _ in range(n):
        sex = 1 if rng.random() < p else 0
        height = rng.normal(synthetic_height_mean(sex), synthetic_height_sd(sex))
        power = rng.normal(synthetic_power_mean(sex), synthetic_power_sd(sex))
        data["sex"].append(sex)
        data["height"].append(height)
        y.append(height + power)
    return pd.DataFrame(data, columns=keys), np.array(y), ["sex"]


def load_scut(
    seed=None,
    data_root=None,
    target="Average",
    protected=None,
    **kwargs,
):
    data_root = _resolve_scut_root(data_root)
    ratings_path = data_root / "ImageExp" / "Selected_Ratings.csv"
    image_root = data_root / "Images"
    if not ratings_path.exists():
        raise FileNotFoundError(f"SCUT ratings not found: {ratings_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"SCUT images not found: {image_root}")

    ratings = pd.read_csv(ratings_path)
    ratings.columns = ratings.columns.str.strip()
    ratings["Filename"] = ratings["Filename"].astype(str).str.strip()
    if target not in ratings.columns:
        raise ValueError(f"SCUT target column '{target}' not found in {ratings_path}")
    ratings = ratings[["Filename", target]].dropna()

    paths = ratings["Filename"].map(lambda filename: image_root / filename)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"SCUT images missing; first missing file: {missing[0]}")

    features = pd.DataFrame({"image_path": paths.map(str)})
    features["sex"] = ratings["Filename"].map(lambda name: 1 if str(name)[1].upper() == "M" else 0).to_numpy()
    features["race"] = ratings["Filename"].map(lambda name: 1 if str(name)[0].upper() == "C" else 0).to_numpy()
    y = ratings[target].to_numpy(dtype=float)
    y = y / np.nanmax(y)
    return features.reset_index(drop=True), y, protected or ["sex", "race"]


def load_scut_sex(seed=None, **kwargs):
    return load_scut(seed=seed, protected=["sex"], **kwargs)


def load_scut_multi(seed=None, **kwargs):
    return load_scut(seed=seed, protected=["sex", "race"], **kwargs)


def _resolve_scut_root(data_root):
    if data_root:
        return Path(data_root)
    candidates = []
    if os.environ.get("SCUT_DATA_ROOT"):
        candidates.append(Path(os.environ["SCUT_DATA_ROOT"]))
    candidates.extend([
        REPO_ROOT / "data" / "scut",
        REPO_ROOT.parent / "Comparable" / "Data",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]
