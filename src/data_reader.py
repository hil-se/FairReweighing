import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def load_dataset(name, seed=None, **kwargs):
    loaders = {
        "community": load_communities_bi,
        "communities": load_communities_bi,
        "community_con": load_communities_con,
        "community-continuous": load_communities_con,
        "insurance": load_insurance,
        "lsac": load_lsac,
        "german": load_german,
        "heart": load_heart,
        "synthetic": load_synthetic,
        "scut": load_scut,
        "scut-fbp5500": load_scut,
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
    features = pd.DataFrame(data.values.astype(np.float32))

    # Column 2 is racepctblack after dropping identifiers; this keeps the paper's Race% setting explicit.
    protected = [2]
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


def load_lsac(seed=None, **kwargs):
    df = pd.read_csv(DATA_DIR / "lawschool.csv")
    df = df.dropna()
    df["race"] = [int(race == 7.0) for race in df["race"]]
    df, _ = train_test_split(df, test_size=0.7, stratify=df["race"], random_state=seed)
    y = np.array(df["ugpa"] / max(df["ugpa"]))
    df = df.drop("ugpa", axis=1)
    df["gender"] = df["gender"].map({"male": 1, "female": 0})
    df_bar = df["bar1"]
    df = df.drop("bar1", axis=1)
    df["bar1"] = [int(grade == "P") for grade in df_bar]
    return df, y, ["race"]


def load_insurance(seed=None, **kwargs):
    data = pd.read_csv(DATA_DIR / "insurance.csv")
    data["sex"] = data["sex"].apply(lambda x: 1 if x == "male" else 0)
    dependent = "charges"
    X = data.drop(dependent, axis=1)
    X["age"] = X["age"] / max(X["age"])
    y = np.array(data[dependent])
    return X, y / max(y), ["age"]


def load_synthetic(n=5000, p=0.7, seed=None, **kwargs):
    rng = np.random.default_rng(seed)
    keys = ["sex", "height"]
    data = {key: [] for key in keys}
    y = []
    for _ in range(n):
        sex = 1 if rng.random() < p else 0
        height = rng.normal(1.65 + 0.1 * sex, 0.1 + 0.05 * sex)
        power = rng.normal(0.5 + 0.1 * sex, 0.1 + 0.05 * sex)
        data["sex"].append(sex)
        data["height"].append(height)
        y.append(height + power)
    return pd.DataFrame(data, columns=keys), np.array(y), ["sex"]


def load_scut(
    seed=None,
    data_root=None,
    target="Average",
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
    return features.reset_index(drop=True), y, ["sex", "race"]


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
