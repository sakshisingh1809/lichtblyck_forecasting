"""Script to calculate the volume deviations in the p2h and gas portfolios."""

#%% IMPORTS.

from dataclasses import dataclass
from typing import Callable, Dict
from scipy.stats import norm
import lichtblyck as lb
import pandas as pd
import numpy as np


@dataclass
class Pf:
    tlp: Callable
    offtake: Dict[str, lb.PfLine] = None
    deviation: lb.PfLine = None
    distr: pd.DataFrame = None
    deviation_per_calmonth: pd.Series = None


# %% TEMPERATURE TO OFFTAKE.

# Temperature weights.
weights = pd.DataFrame(
    {
        "power": [0.1, 0.7, 1.3, 1.5, 1.9, 0.7, 0.3, 0, 0.1, 0.1, 0, 1.3, 1.1, 0, 0],
        "gas": [0.7, 4.0, 13, 29.0, 13.2, 3.6, 2.9, 0, 1.8, 0.4, 0, 9.4, 3.4, 0, 0.1],
    },
    index=range(1, 16),
)  # GWh/a in each zone
weights = weights["gas"] / weights["gas"].sum()

# Expected temperatures.
ti = pd.date_range("2005", "2023", freq="D", tz="Europe/Berlin", closed="left")
tau = (ti - pd.Timestamp("1900-01-01", tz="Europe/Berlin")).total_seconds()
tau = tau / 3600 / 24 / 365.24  # zeit in jahresfractionen seit 1900
t_exp = pd.Series(
    5.84 + 0.0379 * tau + -9.034 * np.cos(2 * np.pi * (tau - 19.366 / 365.24)), ti
)

# Actual temperature.
t = lb.tmpr.hist.tmpr()
t = lb.tmpr.hist.fill_gaps(t)
t_act = t.wavg(weights.values, axis=1)

# Temperature to offtake.
tlp_rh = lb.tlp.power.fromsource(2, spec=571570)
tlp_hp = lb.tlp.power.fromsource(3, spec=54475)
pfs = {
    "p2h": Pf(lambda ts: tlp_rh(ts) + tlp_hp(ts)),
    "gas": Pf(lb.tlp.gas.D14(kw=1000000)),
}
# quick visual check
for label, pf in pfs.items():
    print(label)
    lb.tlp.plot.vs_time(pf.tlp)
    lb.tlp.plot.vs_t(pf.tlp)

# %% PER MONTH: EXPECTED AND ACTUAL OFFTAKE, AND DEVIATION.

for label, pf in pfs.items():
    # offtake.
    pf.offtake = {}
    for key, t in (("act", t_act), ("exp", t_exp)):
        s = pf.tlp(t)
        pfl = lb.PfLine({"w": s.astype("pint[MW]")})
        pf.offtake[key] = pfl.asfreq("MS")
    # deviation.
    fraction = pf.offtake["act"].q / pf.offtake["exp"].q
    pf.deviation = fraction.dropna() - 1


# %% CALCULATE DISTRIBUTION PER MONTH-OF-YEAR.


def distr(s):
    s = s.pint.to("").pint.magnitude  # remove any factors and keep float
    mu = s.mean()
    sigma = s.std()
    return pd.Series(
        {"mean": mu, "std": sigma, "q05": s.quantile(0.05), "q95": s.quantile(0.95)}
    )


for label, pf in pfs.items():
    pf.distr = pf.deviation.groupby(lambda ts: ts.month).apply(distr)
    pf.distr = pf.distr.unstack()

# %% COMBINE TO SINGLE RESULT.

result = pd.DataFrame(
    {
        (label, column): s
        for label, pf in pfs.items()
        for column, s in pf.distr.items()
        if column in ["q05", "q95"]
    }
)
# %%
