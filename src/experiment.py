import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from data_reader import load_communities_con, load_insurance, load_lsac, load_german, load_heart, \
    load_communities_bi, load_synthetic
from density_balance import DensityBalance
from metrics import Metrics


class Experiment():

    def __init__(self, data="Community", regressor="Linear", balance="None", density_model="Neighbor"):
        datasets = {"Community": load_communities_bi, "Insurance": load_insurance,
                    "LSAC": load_lsac, "German": load_german,
                    "Heart": load_heart, "Community_Con": load_communities_con, "Synthetic": load_synthetic}
        regressors = {"SVR": SVR(kernel="linear"), "Linear": LinearRegression(positive=True),
                      "Logistic": LogisticRegression(), "DT": DecisionTreeRegressor(max_depth=8),
                      "RF": RandomForestClassifier()}
        self.X, self.y, self.protected = datasets[data]()
        self.regressor = regressors[regressor]
        self.balance = balance
        self.density_model = density_model
        self.preprocessor = None

    def run(self):
        self.train_test_split()
        self.preprocess(self.X_train)
        X_train = self.preprocessor.transform(self.X_train)
        X_test = self.preprocessor.transform(self.X_test)
        y_train = self.y_train

        y_test = self.y_test
        if self.balance == "None":
            sample_weight = None
        else:
            sample_weight = DensityBalance(model=self.density_model).weight(self.X_train[self.protected].to_numpy(),
                                                                            np.transpose([self.y_train]),
                                                                            treatment=self.balance)
        self.regressor.fit(X_train, y_train, sample_weight)
        result = self.test(X_test, y_test)
        return result

    def preprocess(self, X):
        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)

        numerical_columns = numerical_columns_selector(X)
        categorical_columns = categorical_columns_selector(X)

        categorical_preprocessor = OneHotEncoder(handle_unknown='ignore')
        numerical_preprocessor = StandardScaler()
        self.preprocessor = ColumnTransformer([
            ('OneHotEncoder', categorical_preprocessor, categorical_columns),
            # ('StandardScaler', numerical_preprocessor, numerical_columns)
        ], remainder='passthrough')
        self.preprocessor.fit(X)

    def test(self, X, y):
        y_pred = self.regressor.predict(X)
        Theta = np.linspace(0, 1.0, 41)
        m = Metrics(y, y_pred)
        result = {
            "MSE": m.mse(),
            # "RMSE": m.rmse(),
            # "MAE" : m.mae(),
            "R2": m.r2(),
            # "Accuracy": m.accuracy(),
            # "F1": m.f1(),
        }
        for key in self.protected:
            # acc_joint, acc_margin, ratio = m.r_sep(np.array(self.X_test[key]))
            # result["Acc_joint_" + str(key)] = acc_joint
            # result["Acc_margin_" + str(key)] = acc_margin
            ratio = m.r_sep(np.array(self.X_test[key]))
            ratio_a = m.r_sep_a(np.array(self.X_test[key]))
            result["Ratio_" + str(key)] = ratio
            # result["Ratio_a_" + str(key)] = ratio_a
            # result["DP_" + str(key)] = m.DP(np.array(self.X_test[key]))
            # result["AOD_" + str(key)] = m.AOD(np.array(self.X_test[key]))
            # result["gAOD_" + str(key)] = m.gAOD(np.array(self.X_test[key]))
            # result["cAOD_" + str(key)] = m.cAOD(np.array(self.X_test[key]))
            #
            # result["EOD_" + str(key)] = m.EOD(np.array(self.X_test[key]))
            # result["gEOD_" + str(key)] = m.gEOD(np.array(self.X_test[key]))
            # result["cEOD_" + str(key)] = m.cEOD(np.array(self.X_test[key]))

            # result["GDP_" + str(key)] = m.GDP(np.array(self.X_test[key]))
            result["DP_" + str(key)] = m.DP_disp(self.X_test[key], Theta)
            # result["BGL_mse_" + str(key)] = m.bgl_mse(self.X_test[key])
            # result["BGL_mae_" + str(key)] = m.bgl_mae(self.X_test[key])
            result["Con_Indi_" + str(key)] = m.convex_individual(self.X_test[key])
            # result["Con_Grp_" + str(key)] = m.convex_group(self.X_test[key])
        return result

    def train_test_split(self, train_ratio=0.5):
        n = len(self.y)
        train_ind = list(np.random.choice(range(n), int(n * train_ratio), replace=False))
        test_ind = list(set(range(n)) - set(train_ind))
        self.X_train = self.X.iloc[train_ind]
        self.X_test = self.X.iloc[test_ind]
        self.y_train = self.y[train_ind]
        self.y_test = self.y[test_ind]
