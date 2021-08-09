"""
Script to calculate the input for Gilbert's credit risk calculations.
"""


# %% Imports.

import lichtblyck as lb
import pandas as pd

# %% Temperatures / heating-degree-days.

t = lb.tmpr.hist.tmpr()
t = t[t.index >= "2000"]
hdd = t.applymap(lambda x: max(18 - x, 0)).mul(t.duration / 24, axis=0)
hdd_per_month_and_zone = lb.changefreq_sum(hdd, "MS")
weights = lb.tmpr.weights()
hdd_per_month = lb.wavg(hdd_per_month_and_zone, weights["power"], axis=1)

hdd_per_calmonth = (
    hdd_per_month.groupby(lambda ts: ts.month)
    .apply(lambda df: {"mean": df.mean(), "std": df.std()})
    .unstack()
    .rename_axis(index="calmonth")
)

# %% Offtake.

tlp = lb.tlp.power.fromsource(1, spec=1)
wo_per_day_and_zone = pd.DataFrame({key: tlp(s) for key, s in t.items()})
wo_per_day = lb.wavg(wo_per_day_and_zone, weights["power"], axis=1)
wo_per_month = lb.changefreq_avg(wo_per_day, "MS")
qo_per_month = wo_per_month.mul(wo_per_month.duration, axis=0)
qo_per_calmonth = (
    qo_per_month.groupby(lambda ts: ts.month).mean().rename_axis(index="calmonth")
)
qo_per_calmonth_fraction = qo_per_calmonth / qo_per_calmonth.sum()
