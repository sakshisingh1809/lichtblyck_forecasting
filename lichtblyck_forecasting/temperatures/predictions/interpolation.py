from matplotlib import axis
import lichtblyck_forecasting as lf
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":

    t = lf.temperatures.historic.tmpr()
    print(t.isnull().sum())  # total missing values for each climatezones
    print(t.isnull().sum().sum())  # total number of missing values

    percent_missing = np.round(t.isnull().sum() * 100 / len(t), 0)
    missing_values = pd.DataFrame({"percent_missing": percent_missing})
    missing_values.sort_values("percent_missing", ascending=False, inplace=True)
    print(missing_values)
