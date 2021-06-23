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
df = df.rename(columns={"kalk_jpv": "qo_step0", "jpv_abrechnung": "qo_stepI"})
df.qo_step0 /= 1000
df.qo_stepI /= 1000
df["tolused"] = df.qo_stepI / df.qo_step0 - 1

#%% Filter.

mask = (
    (df.medium == "Strom")
    & ((df.status == "in Belieferung") | (df.status == "beendet"))
    & (df.qo_step0 != df.qo_stepI)
    & (df.qo_step0 > 0)
    & (df.qo_stepI > 0)
    & ~df.kunden_name.str.lower().str.contains("tesla")
    & ~df.kunden_name.str.lower().str.contains("eneco")
    & (df.kalk_datum >= "2015")
)

df_all = df[mask]
# %% Visualize.


def plot_histogram(df, cutoff=4):
    fig, ax = plt.subplots(
        1, 2, figsize=(10, 5), sharey=True, gridspec_kw={"width_ratios": [5, 1]}
    )
    fig.tight_layout()
    ax[0].hist(df.tolused, bins=np.linspace(-1, cutoff, 200))
    ax[1].bar(["", f">{cutoff:.0%}", " "], [0, sum(df.tolused > cutoff), 0], 0.1)
    ax[0].xaxis.label.set_text("Needed tolerance band")
    ax[0].xaxis.set_major_formatter("{:.0%}".format)
    ax[0].yaxis.label.set_text("Number of customers")


# %% Some statistics
cutoff = 4  # draw until 400%
cols = [ #columns of interes
    "kunden_name",
    "status",
    "abnahme_lieferbeginn",
    "abnahme_lieferende",
    "kalk_typ",
    "kalk_datum",
    "qo_step0",
    "qo_stepI",
    "tolused",
]
plot_histogram(df_all, cutoff)
df_all.tolused.describe()
df_all[df_all.tolused == -1][cols].sample(10).round(2)
df_all[df_all.tolused > cutoff][cols].sample(10).round(2)

# %%
large = df[df.qo_step0 > 3000]
plot_histogram(large)


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
