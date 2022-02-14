#%%


import pandas as pd
import lichtblyck as lb
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm


df = pd.read_excel("rawdata.xlsx", "rawdata")

df.columns = [c.lower() for c in df.columns]
df = df.reindex(df.kunden_name.dropna().index)
cols = [col for col in df.columns if "jpv" in col]
df[cols] /= 1000  # MWh
df["tolused"] = df.jpv_abrechnung / df.kalk_jpv - 1

#%% Filter to keep relevant data points.

mask = (
    (df.medium == "Strom")
    & ((df.status == "in Belieferung") | (df.status == "beendet"))
    & ~df.kunden_name.str.lower().str.contains("tesla")
    & ~df.kunden_name.str.lower().str.contains("eneco")
    & (df.kalk_datum >= "2010")
    & (df.abnahme_lieferende.dt.year < 2020)
)

df = df[mask]

# %% Find characteristics of the data.

# General information.
df.describe()
#   64570 rows.


# Quantities.

# . No volume in calculation:
df[df.kalk_jpv.isna() | (df.kalk_jpv == 0)]
#   8273 rows (12%)
# . No volume in realisation:
df[df.jpv_abrechnung.isna() | (df.jpv_abrechnung == 0)]
#   1531 rows (2%)

# Kalk_id.

# . Number of distinct customers used on a single calculation.
cust_count = (
    df.groupby("kalk_id").apply(lambda df: len(df.kunden_id.unique())).sort_values()
)
# . . 16014 calculations. 304 have >1 customers. 3 have >100 customers:
cust_count[(cust_count > 1)]
cust_count[(cust_count > 100)]
# . . Let's see the calculations with most distinct customers:
df[df.kalk_id == cust_count.index[-1]]
df[df.kalk_id == cust_count.index[-2]]

# . Number of rows with delivery date before calculation date.
#   (is this a problem?)
reverse = (
    df.groupby("kalk_id")
    .apply(lambda df: sum(df.abnahme_lieferbeginn < df.kalk_datum))
    .sort_values()
)
# . . 16014 calculations. 145 have >1 reverse rows. 2 have >100 reverse rows:
reverse[(reverse > 1)]
# . . Let's see the calculation with most reverse rows:
df[df.kalk_id == reverse.index[-1]]
df[df.kalk_id == reverse.index[-2]]


# Buendel_id.

# . Quality check: einzelcalculations that have buendelid. Should be 0? Is 1354:
df[~df.buendel_id.isna() & (df.kalk_typ == "Einzelkalkulation")]
# . Quality check: buendel/finialcalculations that do not have buendelid. Should be 0? is 8275:
df[df.buendel_id.isna() & (df.kalk_typ != "Einzelkalkulation")]

# . Number of distinct customers with same buendel.
cust_count = (
    df.groupby("buendel_id").apply(lambda df: len(df.kunden_id.unique())).sort_values()
)
# . . 551 buendels. 388 have > 1 customer. 4 have >100 customers:
cust_count[(cust_count > 1)]
cust_count[(cust_count > 100)]
# . . Let's see the buendels with most distinct customers:
df[df.buendel_id == cust_count.index[-1]]
df[df.buendel_id == cust_count.index[-2]]


big = df.groupby(["buendel_id", "kalk_id"]).apply(
    lambda df: df[["kalk_jpv", "jpv_abrechnung"]].sum()
)


# What I do not understand.
# . One kalk_id can have >1 buendel_id


# ----------------------------------------- End of exploration --------

# %% Visualization functions.


def plot_histogram_percent(df, cutoff=None):
    s = df.tolused
    avg = df.jpv_abrechnung["sum"].sum() / df.kalk_jpv["sum"].sum() - 1
    label = f"n = {len(s)}\navg = {avg:.1%}"
    if not cutoff:
        cutoff = s.quantile(0.98)
    fig, ax = plt.subplots(
        1, 2, figsize=(10, 5), sharey=True, gridspec_kw={"width_ratios": [5, 1]}
    )
    fig.tight_layout()
    ax[0].hist(s, bins=np.linspace(-1, cutoff, 200), label=label)
    ax[1].bar(["", f">{cutoff:.0%}", " "], [0, sum(s > cutoff), 0], 0.1)
    ax[0].xaxis.label.set_text("Tolerance band used [%]")
    ax[0].xaxis.set_major_formatter("{:.0%}".format)
    ax[0].yaxis.label.set_text("Number of customers")
    ax[0].legend()
    return fig, ax


def plot_histogram_absolute(df, cutofflo=None, cutoffhi=None):
    s = df.jpv_abrechnung["sum"] - df.kalk_jpv["sum"]
    label = f"n = {len(s)}\navg = {s.mean():.0f} MWh"
    if not cutofflo or not cutoffhi:
        cutofflo, cutoffhi = [10 * (s.quantile(q) // 10) for q in [0.02, 0.98]]
    fig, ax = plt.subplots(
        1, 3, figsize=(10, 5), sharey=True, gridspec_kw={"width_ratios": [1, 5, 1]}
    )
    fig.tight_layout()
    ax[1].hist(s, bins=np.linspace(cutofflo, cutoffhi, 200), label=label)
    ax[0].bar(["", f"<{cutofflo:.0f}", " "], [0, sum(s < cutofflo), 0], 0.1)
    ax[2].bar(["", f">{cutoffhi:.0f}", " "], [0, sum(s > cutoffhi), 0], 0.1)
    ax[1].xaxis.label.set_text("Tolerance band used [MWh]")
    ax[0].yaxis.label.set_text("Number of customers")
    ax[1].legend()
    return fig, ax


def plot_scatter_percent(df, cutoff=None):
    s = df.tolused
    avg = df.jpv_abrechnung["sum"].sum() / df.kalk_jpv["sum"].sum() - 1
    label = f"n = {len(s)}\navg = {avg:.1%}"
    if not cutoff:
        cutoff = s.quantile(0.98)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.tight_layout()
    ax.scatter(df.kalk_jpv, s, 2, label=label)
    ax.xaxis.label.set_text("Kalk_jpv [MWh]")
    ax.yaxis.set_major_formatter("{:.0%}".format)
    ax.yaxis.label.set_text("Tolerance band used [%]")
    ax.set_ylim([-1, cutoff])


def plot_scatter_absolute(df, cutofflo=None, cutoffhi=None):
    s = df.jpv_abrechnung["sum"] - df.kalk_jpv["sum"]
    label = f"n = {len(s)}\navg = {s.mean():.0f} MWh"
    if not cutofflo or not cutoffhi:
        cutofflo, cutoffhi = (100 * (s.quantile(q) // 100) for q in (0.02, 0.98))
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.tight_layout()
    ax.scatter(df.kalk_jpv, s, 1.5, label=label)
    ax.xaxis.label.set_text("Kalk_jpv [MWh]")
    ax.yaxis.set_major_formatter("{:.0f}".format)
    ax.yaxis.label.set_text("Tolerance band used [MWh]")
    ax.set_ylim([cutofflo, cutoffhi])


# %% Dataset to work with.

# One line per kalkulation
df2 = (
    df.reset_index()
    .groupby("kalk_id")
    .aggregate(
        {
            "index": "count",
            "jpv_abrechnung": "sum",
            "jpv_prognose": "sum",
            "kalk_jpv": "sum",
            "kalk_typ": "first",
            "buendel_id": ["nunique", lambda x: list(np.unique(x))],
            "kunden_name": ["nunique", lambda x: list(np.unique(x))],
            "abnahme_lieferbeginn": ["nunique", lambda x: list(np.unique(x))],
            "abnahme_lieferende": ["nunique", lambda x: list(np.unique(x))],
            "kalk_datum": "first",  # are all same
        }
    )
)
df2["tolused"] = df2.jpv_abrechnung / df2.kalk_jpv - 1

# Remove kalkulations of which no update has taken place yet (i.e., jpv_abrechnung == kalk_jpv)
mask = (df2.jpv_abrechnung != df2.kalk_jpv) & (df2.kalk_jpv > 0)
df2 = df2[mask["sum"]]

# %% Tolerance band of ALL kalkulations.

cutoff = 4  # draw until 400%
cols = [  # columns of interes
    "kunden_name",
    "abnahme_lieferbeginn",
    "abnahme_lieferende",
    "kalk_typ",
    "kalk_datum",
    "kalk_jpv",
    "jpv_abrechnung",
    "tolused",
]
# Tolerance band.
plot_histogram_percent(df2, cutoff)
plot_histogram_absolute(df2)
# Tolerance band vs customer size.
plot_scatter_percent(df2, cutoff)
plot_scatter_absolute(df2, -1500, 1500)

# Samples of situations with -100% and >400% tolband use.
df2[df2.tolused == -1][cols].sample(10).round(2)
df2[df2.tolused > 4][cols].sample(10).round(2)

# Total tolerance band, over all kalkulations.
volavg_tolused = (df2.tolused * df2.kalk_jpv["sum"]).sum() / (df2.kalk_jpv["sum"]).sum()
print(f"All kalkulations: avg. tolerance band used: {volavg_tolused:.1%}.")


# %% Tolerance band of small customers.

small = df2[df2.kalk_jpv["sum"] < 3000]
plot_histogram_percent(small, cutoff)
plot_scatter_percent(small)
plot_histogram_absolute(small)

# Conclusions:
# a) Enough customers to have a 'functioning' portfolio effect.
# b) But: customer consumption is consistently overestimated so that -11.2% tolerance
#    band is needed in practice.
# So: small customers DO need tolerance band, even if portfolio effects are considered.
# Needed premium depends on risk aversion, maturity and possible price deviations.
# More analyses needed to determine adecuate value [Eur/MWh].

# %% Tolerance band of large customers.

big = df2[df2.kalk_jpv["sum"] >= 3000]
plot_histogram_percent(big, cutoff)
plot_histogram_absolute(big)
plot_scatter_percent(big)


# %% -------------------------------------- Theoretical analyses -------

# %% How sample size changes uncertainty in average tolerance band

# Simulate.
samplesizes = [3, 10, 30, 100, 300, 1000, 3000, 10000]
tol = pd.DataFrame([], columns=["std", "mean"], index=samplesizes)
for samplesize in samplesizes:
    av_tol = []
    for sim in range(3000):  # 3000 simulations should be enough
        sample = df_all[["qo_step0", "qo_stepI"]].sample(samplesize, replace=True)
        qo_step0, qo_stepI = sample[["qo_step0", "qo_stepI"]].sum()
        av_tol.append(qo_stepI / qo_step0 - 1)
    av_tol = pd.Series(av_tol)
    tol.loc[samplesize, "mean"] = av_tol.mean()
    tol.loc[samplesize, "std"] = av_tol.std()

# Visualize.
fig, ax = plt.subplots()
tol.plot(logx=True, ax=ax)
ax.xaxis.label.set_text("Number of customers in PF")
ax.yaxis.set_major_formatter("{:.0%}".format)
ax.yaxis.label.set_text(
    "Resulting average needed tolerance band\nand standard dev in the average"
)
# %%
