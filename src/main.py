import numpy as np
import pandas as pd

from demos import cmd
from experiment import Experiment


def german(density_model='Neighbor', repeat=30):
    data = "German"
    regressor = "Logistic"
    treatments = ["None", "Reweighing"]
    results = []
    for treatment in treatments:
        result = run(data=data, regressor=regressor, balance=treatment,
                     density_model=density_model, repeat=repeat)
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv("../result/" + data + "_" + density_model + ".csv", index=False)


def heart(density_model='Neighbor', repeat=30):
    data = "Heart"
    regressor = "Logistic"
    treatments = ["None", "Reweighing"]
    results = []
    for treatment in treatments:
        result = run(data=data, regressor=regressor, balance=treatment,
                     density_model=density_model, repeat=repeat)
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv("../result/" + data + "_" + density_model + ".csv", index=False)


def community(density_model='Neighbor', repeat=30):
    data = "Community"
    regressor = "Linear"
    treatments = ["None", "Reweighing"]
    results = []
    for treatment in treatments:
        result = run(data=data, regressor=regressor, balance=treatment, density_model=density_model, repeat=repeat)
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv("../result/" + data + "_" + density_model + ".csv", index=False)


def community_con(density_model='Neighbor', repeat=30):
    data = "Community_Con"
    regressor = "Linear"
    treatments = ["None", "Reweighing"]
    results = []
    for treatment in treatments:
        result = run(data=data, regressor=regressor, balance=treatment, density_model=density_model, repeat=repeat)
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv("../result/" + data + "_" + density_model + ".csv", index=False)


def lsac(density_model='Neighbor', repeat=30):
    data = "LSAC"
    regressor = "Linear"
    treatments = ["None", "Reweighing"]
    results = []
    for treatment in treatments:
        result = run(data=data, regressor=regressor, balance=treatment,
                     density_model=density_model, repeat=repeat)
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv("../result/" + data + "_" + density_model + ".csv", index=False)


def insurance(density_model='Neighbor', repeat=30):
    data = "Insurance"
    regressor = "Linear"
    treatments = ["None", "Reweighing"]
    results = []
    for treatment in treatments:
        result = run(data=data, regressor=regressor, balance=treatment, density_model=density_model, repeat=repeat)
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv("../result/" + data + "_" + density_model + ".csv", index=False)


def synthetic(density_model='Neighbor', repeat=10):
    data = "Synthetic"
    regressor = "Linear"
    treatments = ["None", "Reweighing", "GroundTruth"]
    results = []
    for treatment in treatments:
        result = run(data=data, regressor=regressor, balance=treatment,
                     density_model=density_model, repeat=repeat)
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv("../result/" + data + "_" + density_model + ".csv", index=False)


def run(data="Community", regressor="Linear", balance="Reweighing", density_model="Neighbor", repeat=30):
    runner = Experiment(data=data, regressor=regressor, density_model=density_model, balance=balance)
    results = []
    for i in range(repeat):
        result = runner.run()
        results.append(result)
    df = pd.DataFrame(results)
    output = {"Treatment": balance}
    for key in df.keys():
        output[key] = "%.3f +/- %.3f" % (np.mean(df[key]), np.std(df[key]))
    return output


if __name__ == "__main__":
    eval(cmd())
