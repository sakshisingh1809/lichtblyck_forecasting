import pandas as pd

filenames = {
    "gas": "gas_offtakescalingfactors_per_calmonth_and_degC.csv",
    "p2h": "p2h_offtakescalingfactors_per_calmonth_and_degC.csv",
}

ss = {}
for pf, path in filenames.items():
    table = pd.read_csv(path, index_col=0)
    sensitivity = table["1 degC"] - 1
    ss[pf] = sensitivity

sensitivity = pd.DataFrame(ss)
