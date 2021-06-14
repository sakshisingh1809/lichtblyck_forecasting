"""
Calculate the influence on PNL due to deviations from plan volume
(i.e., needed tolerance band) in past few years.
"""

#%%

from pathlib import Path
import lichtblyck as lb
import pandas as pd
import numpy as np
from lichtblyck.prices.utils import is_peak_hour
from scipy.stats import norm

#%% Prices: yearly base, peak, offpeak prices.

fut_a = lb.prices.power_futures("a")  # year prices, many trading days
fut_m = lb.prices.power_futures("m")  # month prices, many trading days
# spot = lb.prices.power_spot()
# spot_bpo = lb.prices.convert.tseries2bpoframe(spot, "AS")


def monthprice_L(monthprices):
    mask = (monthprices.anticipation > pd.Timedelta(days=26)) & (
        monthprices.anticipation < pd.Timedelta(days=40)
    )
    df = monthprices[mask].dropna()
    if not df.empty:
        return df.iloc[-1]
    return pd.Series([], dtype=np.float32)


# Month prices at single trading day.
fut_m_bpo_stepL = (
    fut_m.groupby(level=0).apply(monthprice_L).unstack().resample("MS").asfreq()
)
# Year prices at end of step L (= after LT period).
fut_a_bpo_stepL = lb.prices.convert.bpoframe2bpoframe(fut_m_bpo_stepL, "AS").iloc[1:-1]

#%% Planned consumption.

# path = Path("lichtblyck") / "scripts" / "2021-06 toleranz" / "profile.xlsx"
path = Path("profile.xlsx")
wo_step0 = pd.read_excel(path, "Tabelle1", header=4, usecols="B:C", index_col=0)
wo_step0 = lb.tools.set_ts_index(wo_step0, bound="right")["MW"]
qo_step0 = wo_step0 * wo_step0.duration
# Decompose in peak and offpeak
qo_step0 = (
    qo_step0.groupby(is_peak_hour).sum().rename({True: "q_peak", False: "q_offpeak"})
)

#%% Tolerance band.


def pnl_tolband(anticipation, tolerance, delivery_year):
    # At acquisition.
    ts_left = pd.Timestamp(f"{delivery_year}-01-01 00:00:00", tz="Europe/Berlin")
    ts_trade_latest = ts_left - anticipation + pd.Timedelta(days=3)
    ts_trade_earliest = ts_left - anticipation - pd.Timedelta(days=3)
    mask = (
        (fut_a.index.get_level_values("ts_left") == ts_left)
        & (fut_a.index.get_level_values("ts_left_trade") < ts_trade_latest)
        & (fut_a.index.get_level_values("ts_left_trade") > ts_trade_earliest)
    )
    df = fut_a[mask].dropna()
    if df.empty:
        return None
    pu_step0 = df[["p_peak", "p_offpeak"]].mean()

    # 3-4 weeks before start of delivery month.
    pu_stepL = fut_a_bpo_stepL.loc[ts_left][["p_peak", "p_offpeak"]]
    qo_stepL = (1 + tolerance) * qo_step0

    # Tolerance band.
    delta_pu_stepL = pu_stepL - pu_step0
    delta_qo_stepL = qo_stepL - qo_step0
    r_tol = sum(
        [
            delta_pu_stepL[f"p_{prod}"] * delta_qo_stepL[f"q_{prod}"]
            for prod in ["peak", "offpeak"]
        ]
    )
    p_tol = r_tol / sum(qo_stepL)

    dic = {}  #'p': {}, 'q': {}, 'r': {}} # return dictionary
    for key, p, q in [
        ("0", pu_step0, qo_step0),
        ("L", pu_stepL, qo_stepL),
        ("delta_L", delta_pu_stepL, delta_qo_stepL),
    ]:
        dic[key] = {
            **p[["p_peak", "p_offpeak"]],
            **q[["q_peak", "q_offpeak"]],
            "q": sum(q[["q_peak", "q_offpeak"]]),
        }
    dic["tol"] = {"r": r_tol, "p": p_tol}
    return dic


records = []
for anticipation in [pd.Timedelta(days=180), pd.Timedelta(days=180 + 365)]:
    for tolerance in [-0.95, -0.5, -0.2, 0, 0.2, 0.5, 1, 1.5, 2]:
        for delivery_year in range(2003, 2021):
            result = pnl_tolband(anticipation, tolerance, delivery_year)
            if result is not None:
                result = {
                    **result,
                    "index": {
                        "anticipation": anticipation,
                        "tolerance": tolerance,
                        "ts_left": delivery_year,
                    },
                }
                records.append(result)
records = [
    {(i, j): record[i][j] for i in record.keys() for j in record[i].keys()}
    for record in records
]
results = pd.DataFrame.from_records(records)
results.columns = pd.MultiIndex.from_tuples(results.columns)
results.index = pd.MultiIndex.from_frame(results["index"])
results = results.drop(columns="index")


# %% Visualise.

# Average PNL impact.
tol_mean = results["tol"].mean(level=(0, 1))
print(tol_mean)
# Positive tolerance band gives negative (average) tolerance band costs. Therefore, prices 
# must have fallen during the period under consideration. Let's verify:
selection = results.loc[pd.IndexSlice[:, results.index.get_level_values(1)[0], :]]
for prod in ["p_peak", "p_offpeak"]:
    selection[[("0", prod), ("L", prod)]].unstack(0).plot(figsize=(16, 10))

# %% Tolerance band needed.

# Find standard deviation and (80%-quantile - mean) from the data.
std_factor = norm().ppf(0.8)
tol_distr = results["tol"].groupby(level=(0, 1)).std() * std_factor

# Compare with sales margin.
p_sales_margin = 10  # Eur/MWh
df = lb.core.functions.add_header(tol_distr, "tol")
df[("margin", "r")] = (
    results[("L", "q")].groupby(level=(0, 1)).mean() * p_sales_margin
)
df[("margin", "p")] = p_sales_margin
# %%
