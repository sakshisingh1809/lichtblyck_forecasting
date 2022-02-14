"""
Compare historic temperatures for a given climate zone with the model prediction. 
Esp. for December.
"""

#%%

import lichtblyck as lb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

climate_zone = "t_3"  # t_3 = hamburg, t_4 = potsdamm

# %% HISTORIC DATA
hist = lb.tmpr.hist.tmpr(True)[climate_zone]

# %% MODEL

a0, a1, a2, a3 = 6.693, 0.029, -8.389, 21.052

ti = pd.date_range(
    "1992-01-01", "2022-01-01", freq="D", tz="Europe/Berlin", closed="left"
)
ti0 = pd.Timestamp("1900-01-01", tz="Europe/Berlin")
tau = (ti - ti0).total_seconds() / 3600 / 24 / 365.24  # years since ti0
model = pd.Series(
    a0 + a1 * tau + a2 * np.cos(2 * np.pi * (tau - a3 / 365.24)),
    index=ti,
)

# %% TEMPERATURES

# Calculate.

t = pd.DataFrame({"hist": hist, "model": model})
avg = t.groupby(lambda ts: (ts.year, ts.month)).mean()
avg.index = pd.MultiIndex.from_tuples(avg.index, names=("year", "month"))
for d in (5, 10, 15):
    avg[f"prev{d}"] = (
        avg["hist"].groupby("month").rolling(d, closed="left").mean().droplevel(0)
    )
avg = avg.dropna()

# Plot.

fig, axes = plt.subplots(3, 4, sharex=True, sharey=False, figsize=(15, 10))
axes = axes.flatten()

colors = {
    "hist": "gray",
    "model": "green",
    "prev5": "red",
    "prev10": "purple",
    "prev15": "blue",
}

for (m, df), ax in zip(avg.groupby("month"), axes):
    ax.set_title(f"Month: {m}")
    for name, s in df.droplevel(1).items():
        ax.plot(s, c=colors.get(name, "gray"), label=name)
    if m == 1:
        ax.legend()

# %% WHICH WAS BETTER?

# Calculate.

series = {"model": avg.model - avg["hist"]}
for d in (5, 10, 15):
    series[f"prev{d}"] = avg[f"prev{d}"] - avg["hist"]
dev = pd.DataFrame(series)
meanstd = dev.groupby("month").apply(lambda df: df.describe().loc[("mean", "std"), :])

# Plot.

fig, axes = plt.subplots(3, 4, sharey=True, figsize=(15, 10))
axes = axes.flatten()

for (m, df), ax in zip(dev.groupby("month"), axes):
    ax.set_title(f"Month: {m}")
    sns.stripplot(ax=ax, data=df, size=2)


fig, axes = plt.subplots(3, 4, sharey=True, figsize=(15, 10))
axes = axes.flatten()

for (m, df), ax in zip(dev.groupby("month"), axes):
    ax.set_title(f"Month: {m}")
    ax.axhline(0, c="gray")
    df.plot(ax=ax, kind="box")
