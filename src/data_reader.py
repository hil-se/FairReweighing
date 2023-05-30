import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_german():
    data = pd.read_csv('../data/german_credit_data.csv', index_col=0)
    data = data.dropna()
    data['Sex'] = data['Sex'].apply(lambda x: 1 if x == "male" else 0)
    data['Risk'] = data['Risk'].apply(lambda x: 1 if x == "good" else 0)
    dependent = 'Risk'
    X = data.drop([dependent], axis=1)
    y = np.array(data[dependent])
    protected = ['Sex']
    return X, y, protected


def load_heart():
    data = pd.read_csv('../data/heart.csv')
    data = data.dropna()
    dependent = 'output'
    X = data.drop([dependent], axis=1)
    y = np.array(data[dependent])
    protected = ['sex']
    return X, y, protected


def load_communities_con():
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

    data = pd.read_csv('../data/communities.data', sep=",", names=column_names,
                       na_values="?")

    assert (not data["ViolentCrimesPerPop"].isna().any())

    labels_df = data["ViolentCrimesPerPop"]
    data.drop(columns="ViolentCrimesPerPop", inplace=True)

    data.drop(columns=["state", "county", "community", "communityname", "fold"],
              inplace=True)

    data.fillna(0, inplace=True)

    labels = labels_df.values.astype(np.float32)
    groups = [2]
    features = pd.DataFrame(data.values.astype(np.float32))

    return features, labels, groups


def load_communities_bi():
    df = pd.read_csv('../data/communities.csv')
    df = df.fillna(0)
    B = "racepctblack"
    W = "racePctWhite"
    A = "racePctAsian"
    H = "racePctHisp"
    sens_features = [2, 3, 4, 5]
    df_sens = df.iloc[:, sens_features]

    Y = np.array(df['ViolentCrimesPerPop'])
    df = df.drop('ViolentCrimesPerPop', axis=1)

    maj = majority_pop(df_sens)

    a = maj.map({B: 0, W: 1, A: 0, H: 0})

    df['race'] = a
    protected = ['race']
    df = df.drop(H, axis=1)
    df = df.drop(B, axis=1)
    df = df.drop(W, axis=1)
    df = df.drop(A, axis=1)
    return df, Y, protected


def majority_pop(a):
    B = "racepctblack"
    W = "racePctWhite"
    A = "racePctAsian"
    H = "racePctHisp"
    maj = a.apply(pd.Series.idxmax, axis=1)
    return maj


def load_lsac():
    df = pd.read_csv('../data/lawschool.csv')
    df = df.dropna()
    df['race'] = [int(race == 7.0) for race in df['race']]
    df, _ = train_test_split(df, test_size=0.7, stratify=df['race'])
    y = df['ugpa']
    y = np.array(y / max(y))
    df = df.drop('ugpa', axis=1)
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    df_bar = df['bar1']
    df = df.drop('bar1', axis=1)
    df['bar1'] = [int(grade == 'P') for grade in df_bar]
    protected = ['race']
    return df, y, protected


def load_insurance():
    data = pd.read_csv('../data/insurance.csv')
    data['sex'] = data['sex'].apply(lambda x: 1 if x == "male" else 0)
    dependent = 'charges'
    X = data.drop(dependent, axis=1)
    X['age'] = X['age'] / max(X['age'])
    y = np.array(data[dependent])
    protected = ['age']
    return X, y / max(y), protected
