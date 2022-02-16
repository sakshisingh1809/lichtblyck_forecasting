"""
Compare historic temperatures for a given climate zone with the prediction of several models. 
"""

#%%

import lichtblyck as lb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

climate_zone = "t_4"  # t_3 = hamburg, t_4 = potsdamm, t_5 = essen, t_12 = mannheim

# %% HISTORIC DATA.
hist = lb.tmpr.hist.tmpr(True)[climate_zone]
t = pd.DataFrame({"hist": hist})

# %% ADD MODELS.

ti = pd.date_range(
    "1992-01-01", "2021-01-01", freq="D", tz="Europe/Berlin", closed="left"
)
ti0 = pd.Timestamp("1900-01-01", tz="Europe/Berlin")
tau = (ti - ti0).total_seconds() / 3600 / 24 / 365.24  # years since ti0

models = {}
hiddenname = True  # if True, use 'A', 'B', 'C' instead of real names.

# linear + cos
name = "A" if hiddenname else "simple"
parameters = {
    "t_3": (6.4698e00, 3.1393e-02, -8.3735e00, 2.0892e01),
    "t_4": (3.7320e00, 5.7780e-02, -9.6124e00, 1.6865e01),
    "t_5": (6.326879, 0.039805, -8.065436, 19.783243),
    "t_12": (8.009155, 0.031172, -9.346911, 16.306732),
}
a0, a1, a2, a3 = parameters[climate_zone]
values = a0 + a1 * tau + a2 * np.cos(2 * np.pi * (tau - a3 / 365.24))
models[name] = pd.Series(values, ti)

# linear + fourierseries
name = "B" if hiddenname else "fourier"
parameters = {
    "t_3": (
        6.42082e00,
        3.20586e-02,
        -8.34319e00,
        2.11414e01,
        2.02009e-01,
        1.25303e-01,
        3.62181e-01,
        -3.96979e00,
        -2.44922e-01,
        -1.24842e01,
        2.40667e-01,
        -1.46246e-02,
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
        2.43193e-01,
        -1.46868e-02,
    ),
}
a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = parameters[climate_zone]
values = a0 + a1 * tau
for n, (a, o) in enumerate(((a2, a3), (a4, a5), (a6, a7), (a8, a9), (a10, a11))):
    values += a * np.cos((n + 1) * 2 * np.pi * (tau - o / 365.24))
models[name] = pd.Series(values, ti)

# polynomial (from Jan)
name = "C" if hiddenname else "polynome"
parameters = {
    "t_3": (
        2.87276085717235,
        -0.039674511701999,
        0.00108024791646852,
        1.31044768994608e-06,
        -2.49759991779024e-08,
        -3.66950769755497e-12,
        2.02635998500888e-13,
        -1.02658805564416e-16,
        -6.17637771654661e-19,
        6.10516790492718e-22,
        3.48105849488361e-25,
        -4.3412094191766e-28,
    ),
    "t_4": (
        1.7746456546555,
        -0.0326854208697564,
        0.00129470990902978,
        8.65649579299504e-07,
        -2.96626641968435e-08,
        5.15186237699272e-12,
        2.35734006253894e-13,
        -1.75938753298615e-16,
        -6.75324721679763e-19,
        8.31059708652118e-22,
        2.08650681652274e-25,
        -4.24430582241011e-28,
    ),
}
a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = parameters[climate_zone]
# most important variation.
values = 0
dayofyear = (tau % 1) * 365.24
for n, a in enumerate((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)):
    values += a * dayofyear ** n
# add linear temperature increase
yearsince2023 = (tau // 1) - 123
incperyear = {"t_3": 0.033, "t_4": 0.055}[climate_zone]
values += incperyear * yearsince2023
models[name] = pd.Series(values, ti)

t = pd.concat([t, pd.DataFrame(models)], axis=1)

t = t.dropna()
tavg = t.groupby(lambda ts: (ts.year, ts.month)).mean()
tavg.index = pd.MultiIndex.from_tuples(tavg.index, names=("year", "month"))

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