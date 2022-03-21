"""
Compare historic temperatures for a given climate zone with the prediction of several models. 
"""

#%%

import lichtblyck as lb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

climate_zone = "t_3"  # t_3 = hamburg, t_4 = potsdamm, t_5 = essen, t_12 = mannheim
tlp_typ = "gas"

# %% HISTORIC DATA.
hist = lb.tmpr.hist.tmpr(True)[climate_zone]
t = pd.DataFrame({"hist": hist})

if tlp_typ == "power":
    details = {"bayernwerk_nsp": 900, "bayernwerk_wp": 100}
    tlps = [lb.tlp.power.fromsource(name, spec=spec) for name, spec in details.items()]
else:
    details = {"D14": 50}
    tlps = [lb.tlp.gas.fromsource(name, kw=spec) for name, spec in details.items()]
tlp = lambda t: sum([prof(t) for prof in tlps])
tlp_label = f"{tlp_typ} {' '.join([f'{key}_{val}' for key, val in details.items()])}"


# %% ADD MODELS.

ti = pd.date_range(
    "1992-01-01", "2021-01-01", freq="D", tz="Europe/Berlin", closed="left"
)
ti0 = pd.Timestamp("2000-01-01", tz="Europe/Berlin")
tau = (ti - ti0).total_seconds() / 3600 / 24 / 365.24  # years since ti0

models = {}
hiddenname = False  # if True, use 'A', 'B', 'C' instead of real names.

# linear + cos
name = "A" if hiddenname else "simple"
parameters = {
    "t_3": (9.60842e00, 3.14050e-02, 8.37243e00, 5.51718e-01),
    "t_4": (3.7320e00, 5.7780e-02, -9.6124e00, 1.6865e01),
    "t_5": (6.326879, 0.039805, -8.065436, 19.783243),
    "t_12": (8.009155, 0.031172, -9.346911, 16.306732),
}
a0, a1, a2, a3 = parameters[climate_zone]
values = a0 + a1 * tau + a2 * np.cos(2 * np.pi * (tau - a3))
models[name] = pd.Series(values, ti)

# linear + fourierseries
name = "B" if hiddenname else "fourier"
parameters = {
    "t_3": (
        9.61428e00,
        3.23511e-02,
        8.31630e00,
        5.52208e-01,
        4.22991e-01,
        8.06904e-02,
        3.10878e-01,
        3.17047e-01,
        2.91373e-01,
        8.03333e-02,
    ),
    "t_4": (
        3.66571e00,
        5.84615e-02,
        -9.57494e00,
        1.70087e01,
        7.99438e-02,
        1.16689e-01,
        4.37583e-01,
        -3.15546e00,
        -3.07479e-01,
        -1.04868e01,
    ),
}
a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = parameters[climate_zone]
values = a0 + a1 * tau
for n, (a, o) in enumerate(((a2, a3), (a4, a5), (a6, a7), (a8, a9))):
    values += a * np.cos((n + 1) * 2 * np.pi * (tau - o))
models[name] = pd.Series(values, ti)

t = pd.concat([t, pd.DataFrame(models)], axis=1)

# %% CREATE SOURCE DATAFRAMES.

t = t.dropna()
tavg = t.groupby(lambda ts: (ts.year, ts.month)).mean()
tavg.index = pd.MultiIndex.from_tuples(tavg.index, names=("year", "month"))

tlim = 17
h = tlim - t
h = h.mask(h < 0, 0)
hsum = h.groupby(lambda ts: (ts.year, ts.month)).sum()
hsum.index = pd.MultiIndex.from_tuples(hsum.index, names=("year", "month"))

o = pd.DataFrame({col: tlp(s) for col, s in t.items()})
osum = o.groupby(lambda ts: (ts.year, ts.month)).sum() * 0.25
osum.index = pd.MultiIndex.from_tuples(osum.index, names=("year", "month"))

# %% COMPARE ABSOLUTE VALUES.


colors = {
    "hist": "gray",
    "simple": "orange",
    "fourier": "green",
    "polynome": "purple",
    "A": "orange",
    "B": "green",
    "C": "purple",
}

# Daily values, in one long graph.

title = f"{climate_zone} - Long graph with daily temperature values"
fig, ax = plt.subplots(figsize=(300, 10))
fig.suptitle(title)
for name, s in t.items():
    s.plot(ax=ax, c=colors.get(name, "gray"), label=name)
ax.legend()
fig.tight_layout()
fig.savefig(f"{title}.png")


# Monthly averages.

title = f"{climate_zone} - Monthly temperature averages"
fig, axes = plt.subplots(3, 4, sharey=False, figsize=(15, 10))
fig.suptitle(title)
for (m, df), ax in zip(tavg.groupby("month"), axes.flatten()):
    ax.set_title(f"Month: {m}")
    for name, s in df.droplevel(1).items():
        ax.plot(s, c=colors.get(name, "gray"), label=name)
    if m == 1:
        ax.legend()
fig.savefig(f"{title}.png")


# Heating degree days per month.

title = f"{climate_zone} - Monthly HDDs"
fig, axes = plt.subplots(3, 4, sharey=False, figsize=(15, 10))
fig.suptitle(title)
for (m, df), ax in zip(hsum.groupby("month"), axes.flatten()):
    ax.set_title(f"Month: {m}")
    for name, s in df.droplevel(1).items():
        ax.plot(s, c=colors.get(name, "gray"), label=name)
    if m == 1:
        ax.legend()
fig.savefig(f"{title}.png")

# Offtake per month.

title = f"{climate_zone} - Monthly Offtake [MWh] {tlp_label}"
fig, axes = plt.subplots(3, 4, sharey=False, figsize=(15, 10))
fig.suptitle(title)
for (m, df), ax in zip(osum.groupby("month"), axes.flatten()):
    ax.set_title(f"Month: {m}")
    for name, s in df.droplevel(1).items():
        ax.plot(s, c=colors.get(name, "gray"), label=name)
    if m == 1:
        ax.legend()
fig.savefig(f"{title}.png")


# %% COMPARE ERROR VALUES.

series = {}
for name, s in t.items():
    if name == "hist":
        continue
    series[name] = s - t["hist"]
derror = pd.DataFrame(series)  # daily error values
merror = derror.groupby(lambda ts: (ts.year, ts.month)).mean()  # monthly error values
merror.index = pd.MultiIndex.from_tuples(merror.index, names=("year", "month"))


# Error vs time.


# Daily error, in one long graph.

title = (
    f"{climate_zone} - Long graph with daily temperature error (nearer to 0 is better)"
)
fig, ax = plt.subplots(figsize=(300, 10))
for name, s in derror.items():
    s.plot(ax=ax, c=colors.get(name, "gray"), label=name)
ax.axhline(0, c="gray")
ax.legend()
fig.tight_layout()
fig.savefig(f"{title}.png")


# Monthly error.

title = f"{climate_zone} - Monthly temperature error (nearer to 0 is better)"
fig, axes = plt.subplots(3, 4, sharey=False, figsize=(15, 10))
fig.suptitle(title)
for (m, df), ax in zip(merror.groupby("month"), axes.flatten()):
    ax.set_title(f"Month: {m}")
    ax.axhline(0, c="gray")
    for name, s in df.droplevel(1).items():
        s.plot(ax=ax, c=colors.get(name, "gray"), label=name)
    if m == 1:
        ax.legend()
fig.savefig(f"{title}.png")


# Error distributions as boxplot.


# Daily error, per calendar month, for all years.

title = f"{climate_zone} - Boxplot of error in daily values, grouped by calendar month, for 1992-2020. (nearer to 0 is better)"
fig, axes = plt.subplots(3, 4, sharey=True, figsize=(15, 10))
fig.suptitle(title)
for (m, df), ax in zip(derror.groupby(lambda ts: ts.month), axes.flatten()):
    ax.set_title(f"Month: {m}")
    ax.axhline(0, c="gray")
    df.plot(ax=ax, kind="box")
    # sns.stripplot(ax=ax, data=df, size=1, alpha=0.5)
fig.savefig(f"{title}.png")

# Monthly average error, per calendar month, for all years.

title = f"{climate_zone} - Boxplot of error in monthly average values, grouped by calendar month, for 1992-2020. (nearer to 0 is better)"
fig, axes = plt.subplots(3, 4, sharey=True, figsize=(15, 10))
fig.suptitle(title)
for (m, df), ax in zip(merror.groupby("month"), axes.flatten()):
    ax.set_title(f"Month: {m}")
    ax.axhline(0, c="gray")
    df.plot(ax=ax, kind="box")
    # sns.stripplot(ax=ax, data=df, size=1, alpha=0.5)
fig.savefig(f"{title}.png")


# RMS or error.


# RMS of error in daily values.

rms = lambda df: np.sqrt(np.mean(df ** 2))
rms_derror = derror.groupby(lambda ts: ts.month).apply(rms)
title = f"{climate_zone} - RMS of error in daily average values, grouped by calendar month, for 1992-2020. (smaller is better)"
fig, axes = plt.subplots(figsize=(15, 10))
fig.suptitle(title)
rms_derror.plot.bar(ax=axes, color=colors)
fig.savefig(f"{title}.png")

# RMS of error in monthly averages.

rms_merror = merror.groupby("month").apply(rms)
title = f"{climate_zone} - RMS of error in monthly average values, grouped by calendar month, for 1992-2020. (smaller is better)"
fig, axes = plt.subplots(figsize=(15, 10))
fig.suptitle(title)
rms_merror.plot.bar(ax=axes, color=colors)
fig.savefig(f"{title}.png")

# RMS of error in monthly averages, by calendar year

rms_merror_peryear = merror.groupby("year").apply(rms)
title = f"{climate_zone} - RMS of error in monthly average values, grouped by calendar year. (smaller is better)"
fig, axes = plt.subplots(figsize=(15, 10))
fig.suptitle(title)
rms_merror_peryear.plot.bar(ax=axes, color=colors)
fig.savefig(f"{title}.png")

# %%
