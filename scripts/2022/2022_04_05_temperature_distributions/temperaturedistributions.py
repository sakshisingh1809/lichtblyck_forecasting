"""Calculate how much the actual temperature deviates from the expected temperature."""

import lichtblyck as lb
import pandas as pd

#%%

act = lb.tmpr.hist.tmpr("1990", "2022")
exp = lb.tmpr.norm.tmpr("1990", "2022")

dev = act - exp
dev = dev.drop(columns="t_13")
#%% monthly

monthly_dev = dev.groupby(lambda ts: (ts.year, ts.month)).mean()
monthly_dev.index = pd.MultiIndex.from_tuples(
    monthly_dev.index, names=("year", "month")
)

monthly_dev_2 = monthly_dev.mean(axis=1).dropna()
monthly_dev_2.groupby(level="month").describe()
monthly_dev_2.groupby(level="month").quantile(0.1).mean()
# %%
