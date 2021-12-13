import lichtblyck as lb
from lichtblyck.tools.nits import Q_
import pandas as pd
import numpy as np
from tqdm import tqdm

weights = pd.DataFrame(
    {
        "power": [0.1, 0.7, 1.3, 1.5, 1.9, 0.7, 0.3, 0, 0.1, 0.1, 0, 1.3, 1.1, 0, 0],
        "gas": [0.7, 4.0, 13, 29.0, 13.2, 3.6, 2.9, 0, 1.8, 0.4, 0, 9.4, 3.4, 0, 0.1],
    },
    index=range(1, 16),
)  # GWh/a in each zone
weights = weights["gas"] / weights["gas"].sum()

# Actual temperature.
t = lb.tmpr.hist.tmpr()
t = lb.tmpr.hist.fill_gaps(t)
t_act = t.wavg(weights.values, axis=1)

# Quick visual check: variation per year.
t_permonth = t_act[t_act.index >= "2001"].resample("MS").mean()
t_permonth.index = t_permonth.index.map(lambda ts: (ts.year, ts.month))
t_permonth = t_permonth.unstack().dropna()
t_permonth.loc["mean"] = t_permonth.mean()
t_permonth.loc["std"] = t_permonth.loc[
    :"mean"
].std() 
t_permonth = t_permonth.round(1)

# Calculate offtake at various temperatures.
t_act = t_act[t_act.index >= "2001"]
offtake = {}
# Temperature -> offtake.
tlp = lb.tlp.gas.D14(kw=1000)
# Hypothetic offtake at various deltas from the historic temperatures.
for delta_t in tqdm(range(-10, 11)):
    offtake_permonth = lb.PfLine({"w": tlp(t_act + delta_t)}).changefreq("MS")
    offtake_percalmonth = (
        offtake_permonth.q.groupby(lambda ts: ts.month)
        .apply(np.mean)
        .astype("pint[MWh]")
    )
    offtake[Q_(delta_t, "degC")] = offtake_percalmonth
offtake = pd.DataFrame(offtake)

# Result 1: offtake and sensitivity at actual temperature.

def getresult1(df):
    df = df.droplevel(0)
    res = pd.DataFrame(
        {
            "offtake": df.loc[Q_(0, "degC"), :],
            "sensitivity1": (df[Q_(1, "degC")] - df[Q_(0, "degC")]) / Q_(1, "degC"),
        }
    )
    res["sensitivity2"] = res["sensitivity1"] / res["offtake"]
    return res

result1 = offtake.T.groupby(axis=0, level=0).apply(getresult1)
result1.pint.dequantify()

# Result 2: offtake fractions at various temperature changes.
def getresult2(df):
    return df / df[Q_(0, "degC")]
result2 = offtake.T.transform(getresult2).T
result2.to_csv('gas_offtakescalingfactors_per_calmonth_and_degC.csv')
# 
