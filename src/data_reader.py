import numpy as np
import pandas as pd
from sklearn import preprocessing

def load_german():
    data = pd.read_csv('../data/german_credit_data.csv', index_col=0)
    # Drop columns with missing values
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


def load_synthetic(n=2000, p=0.5):
    # n is the number of data points.
    # 0 <= p <= 1 is the sampling probability of Male (sex=1).
    keys = ["sex", "age", "work_exp", "hair_length"]
    data = {key: [] for key in keys}
    y = []
    for i in range(n):
        rand = np.random.random()
        sex = 1 if rand < p else 0
        age = int(np.random.normal(40, 15))
        if age < 0:
            age = 0
        hair_length = 35 * np.random.beta(2, 2 + 5 * sex)
        work_exp = np.random.poisson(age + 6 * sex) - np.random.normal(20, 0.2)
        if work_exp < 0:
            work_exp = 0
        income = np.random.normal((50000 + 2000 * work_exp) * (1 if sex == 0 else 1.2),
                                  20000 + 2000 * sex + 200 * work_exp)
        data["sex"].append(sex)
        data["age"].append(age)
        data["work_exp"].append(work_exp)
        data["hair_length"].append(hair_length)
        y.append(income)
    X = pd.DataFrame(data, columns=keys)
    y = np.array(y)
    protected = ["sex"]

    return X, y/max(y), protected


def load_communities():
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

    # Drop first five non-predictive attributes
    data.drop(columns=["state", "county", "community", "communityname", "fold"],
              inplace=True)

    data.fillna(0, inplace=True)

    labels = labels_df.values.astype(np.float32)
    groups = [2]
    features = pd.DataFrame(data.values.astype(np.float32))

    return features, labels, groups


def clean_communities_full():
    """
    Extract black and white dominant communities;
    sub_size : number of communities for each group
    """
    df = pd.read_csv('../data/communities.csv')
    df = df.fillna(0)
    B = "racepctblack"
    W = "racePctWhite"
    A = "racePctAsian"
    H = "racePctHisp"
    sens_features = [2, 3, 4, 5]
    df_sens = df.iloc[:, sens_features]

    # creating labels using crime rate
    Y = np.array(df['ViolentCrimesPerPop'])
    df = df.drop('ViolentCrimesPerPop', axis=1)

    maj = majority_pop(df_sens)

    # remap the values of maj
    a = maj.map({B: 0, W: 1, A: 0, H: 0})

    df['race'] = a
    protected = ['race']
    df = df.drop(H, axis=1)
    df = df.drop(B, axis=1)
    df = df.drop(W, axis=1)
    df = df.drop(A, axis=1)
    return df, Y, protected

def majority_pop(a):
    """
    Identify the main ethnicity group of each community
    """
    B = "racepctblack"
    W = "racePctWhite"
    A = "racePctAsian"
    H = "racePctHisp"
    races = [B, W, A, H]
    maj = a.apply(pd.Series.idxmax, axis=1)
    return maj

def load_lsac():
    data = pd.read_csv('../data/bar_pass_prediction.csv')
    data = data[((data.race1 == 'white') | (data.race1 == 'black'))]
    data = data.dropna()
    data['gender'] = data['gender'].apply(lambda x: 1 if x == "male" else 0)
    data['race1'] = data['race1'].apply(lambda x: 1 if x == "white" else 0)
    dependent = 'gpa'
    X = data.drop([dependent, 'ugpa'], axis=1)
    y = np.array(data[dependent])
    protected = ['race1']
    return X, y/max(y), protected

def clean_lawschool_full():
    """

    Use race as the protected feature.
    sub_size : the number of observations to include for each group
    """
    df = pd.read_csv('../data/lawschool.csv')
    df = df.dropna()
    # remove y from df
    y = df['ugpa']
    y = np.array(y / max(y))
    df = df.drop('ugpa', axis=1)
    # convert gender variables to 0,1
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    # add bar1 back to the feature set
    df_bar = df['bar1']
    df = df.drop('bar1', axis=1)
    df['bar1'] = [int(grade == 'P') for grade in df_bar]
    df['race'] = [int(race == 7.0) for race in df['race']]
    protected = ['race']
    return df, y, protected


def load_insurance():
    data = pd.read_csv('../data/insurance.csv')
    data['sex'] = data['sex'].apply(lambda x: 1 if x == "male" else 0)
    data['age'] = [int(age >= 40) for age in data['age']]
    dependent = 'charges'
    X = data.drop(dependent, axis=1)
    y = np.array(data[dependent])
    protected = ['age']
    return X, y/max(y), protected

def load_insurance_con():
    data = pd.read_csv('../data/insurance.csv')
    data['sex'] = data['sex'].apply(lambda x: 1 if x == "male" else 0)
    dependent = 'charges'
    X = data.drop(dependent, axis=1)
    y = np.array(data[dependent])
    protected = ['age']
    return X, y, protected

