"""
Wanted: Delta Price (actual - expected_at_monthstart) vs 
        Delta Quantity (actual - expected_at_monthstart)

Quantity as expected at start of month is not known / stored in system.

Therefore: Delta Price vs Delta Temperature, 
with Delta Temperature = actual - month average.
"""

# %%

import lichtblyck as lb
import pandas as pd
import datetime as dt

# Get situation as realized.
lb.belvis.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")
pf = lb.portfolios.pfstate("power", "B2C_P2H", "2021-12", "2022")

# Get prices before start of month.
qhpfc_L = pd.read_excel(
    "QHPFC_before_month.xlsx",
    header=None,
    names=["date", "time", "p"],
    usecols=[0, 1, 4],
)
qhpfc_L.index = qhpfc_L.apply(
    lambda r: dt.datetime.strptime(f"{r['date']} {r['time']}", "%d.%m.%Y %H:%M:%S"),
    axis=1,
)
qhpfc_L = lb.set_ts_index(qhpfc_L["p"], bound="right")
qhpfc_L = lb.PfLine(qhpfc_L)

# Get temperatures.
tempr = pd.read_excel("Temperaturen.xlsx", header=0, names=["t"], usecols=[2])
tempr.index = pf.asfreq("D").offtake.index
tempr["delta_t"] = tempr.t - tempr.t.mean()

# %% Spot price vs temperature.

import matplotlib.pyplot as plt

delta_p = pf.unsourcedprice - qhpfc_L
i = delta_p.index.map(lambda ts: ts.floor("D"))
delta_t = tempr.delta_t.loc[i]
plt.plot(delta_t, delta_p.p, ".b", alpha=0.2)


# %%

d = pf.asfreq("D")
d.to_clipboard()

# %%

p = lb.PfLine.from_belvis_forwardpricecurve("power", "2021")
# %%
