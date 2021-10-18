"""
Module to calculate the Beschaffungskostenaufschlag.

Abbreviations:
pu = market price
ps = sourced price
rs = sourced cost
ws = sourced power [MW]
qs = sourced volume [MWh]
wo = offtake power [MW]
wq = offtake volume [MWh]
"""

# %% IMPORTS

import lichtblyck as lb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
from pathlib import Path

# %% PREPARATIONS
__file__ = "."

pf = "B2C"  # "LUD" or "B2C"

if pf == "LUD":
    usecols = [0, 1, 4, 6, 8]  # B2C
else:
    usecols = [0, 1, 10, 12, 14]  # Lud

# Get current situation, prepare dataframe.
current = pd.read_excel(
    Path(__file__).parent / "20210804_090841_Zeitreihenbericht_update.xlsx",
    header=1,
    index_col=0,
    usecols=usecols,
    names=["ts_local_right", "pu", "wo", "ws", "rs"],
)
current = lb.set_ts_index(current, bound="right")
current.wo = -current.wo

if pf == "B2C":
    # Correction: the sourced volume is actually for this PF but also for WP and NSP.
    contrib1 = pd.DataFrame({"PK": {"b": 93, "p": 29}, "P2H": {"b": 12, "p": -6}}).T
    contrib1["peak"] = contrib1.p + contrib1.b
    contrib1["offpeak"] = contrib1.b
    contrib1 = contrib1.drop(columns=["p", "b"])
    contrib2 = contrib1.loc["PK", :] / contrib1.sum()
    contrib3 = contrib2.to_frame().T
    contrib3.index = pd.date_range("2022", periods=1, freq="AS", tz="Europe/Berlin")
    contrib4 = lb.prices.convert.bpoframe2tseries(
        contrib3.rename(columns={"peak": "p_peak", "offpeak": "p_offpeak"}),
        current.index.freq,
    )  # (confusing: these are not prices! TODO: change functions)
    current.ws *= contrib4
    current.rs *= contrib4

# Calculate additional information.
current["ps"] = current.rs / (current.ws * current.duration)
current["wu"] = current.wo - current.ws
current["ru"] = current.wu * current.duration * current.pu
current["ro"] = current.rs + current.ru


#%% OFFTAKE

# Get other offtake curves.
def wo_withchurn(churnstart: float = 0, churnend: float = 0.1):
    """Function to calculate prognosis paths.
    churnstart: (excess) churn at start of year, i.e., compared to expected path.
    churnend: (excess) churn at end of year, i.e., compared to expected path."""
    yearfrac = (current.index - current.index[0]) / (
        current.index[-1] - current.index[0]
    )
    churnpath = churnstart * (1 - yearfrac) + churnend * yearfrac
    return current["wo"] * (1 - churnpath)


# Scenarios between which to simulate.
worst = {"churnstart": 0.02, "churnend": 0.1}
best = {"churnstart": -0.02, "churnend": -0.1}


# Simulate actual offtake path, between lower and upper bound (uniform distribution).
def wo_sim():
    howbad = np.random.uniform()
    start = howbad * worst["churnstart"] + (1 - howbad) * best["churnstart"]
    end = howbad * worst["churnend"] + (1 - howbad) * best["churnend"]
    return wo_withchurn(start, end).rename("wo")


# Quick visual check.
fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
ref = lb.changefreq_avg(current.wo, "D")
for sim in range(300):
    w = lb.changefreq_avg(wo_sim(), "D")
    ax[0].plot(w, alpha=0.03, color="k")
    ax[1].plot(w / ref, alpha=0.03, color="k")
ax[0].plot(w, color="k")  # accent on one
ax[1].plot(w / ref, color="k")  # accent on one
ax[0].plot(lb.changefreq_avg(current.wo, "D"), color="r")
ax[1].plot(
    lb.changefreq_avg(current.wo, "D").div(ref, axis=0), color="r",
)
ax[0].yaxis.label.set_text("MW")
ax[1].yaxis.label.set_text("as fraction of expected offtake")
ax[1].yaxis.set_major_formatter("{:.0%}".format)


# %% PRICES

# Get prices with correct expectation value (=average).
p_sims = pd.read_csv(Path(__file__).parent / "MC_NL_HOURLY_PWR.csv")
# . Clean: make timezone-aware
p_sims = p_sims.set_index("deldatetime")
p_sims.index = pd.to_datetime(p_sims.index, format="%d-%m-%Y %H:%M")
p_sims.index.freq = p_sims.index.inferred_freq
# . . The goal: timezone-aware index.
idx = p_sims.index  # the original: 24 hours on each day. This is incorrect.
idx_EuropeBerlin = pd.date_range(idx[0], idx[-1], freq=idx.freq, tz="Europe/Berlin")
# . . The corresponding local datetime.
idx_local = idx_EuropeBerlin.tz_localize(None)
# . . Use local datetime to find correct values for the timezone-aware datetime.
p_sims = p_sims.loc[idx_local, :].set_index(idx_EuropeBerlin)
# . . Rename index
p_sims = lb.set_ts_index(p_sims)

# . Clean: make arbitrage free
p_sims = lb.changefreq_avg(p_sims, current.pu.index.freq)  # correct frequency
factor = (current.pu / p_sims.mean(axis=1)).dropna()  # find correction factor
p_sims = p_sims.multiply(factor, axis=0).dropna()  # apply

# Get a price simulation.
def p_sim():
    i = np.random.randint(len(p_sims.columns))
    return p_sims.iloc[:, i]


# Quick visual check.
fig, ax = plt.subplots(figsize=(16, 10))
for sim in range(300):
    ax.plot(lb.changefreq_avg(p_sim(), "D"), alpha=0.03, color="k")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(lb.changefreq_avg(p_sim(), "D"), color="k")  # accent on one
ax.plot(lb.changefreq_avg(current.pu, "D"), color="r")


# %% OFFER (=INITIAL SITUATION)

# Find 'eventual' sourced volume at offer time, and current (best-guess) offer price.
w_hedge_missing = lb.hedge(current.wu, current.pu, "AS", "val", bpo=True)
w_hedge = current.ws + w_hedge_missing
r_hedge = current.rs + w_hedge_missing * w_hedge_missing.duration * current.pu

offer = current.copy()
offer["ws"] = w_hedge
offer["rs"] = r_hedge
offer["ps"] = offer.rs / (offer.ws * offer.duration)
offer["wu"] = offer.wo - offer.ws
offer["ru"] = offer.wu * offer.duration * offer.pu
offer["ro"] = offer.rs + offer.ru
stepO = pd.Series(dtype=float)
stepO["qs"] = (offer.ws * offer.duration).sum()
stepO["ps"] = offer.rs.sum() / stepO["qs"]
stepO["ro"] = offer.ro.sum()
stepO["qo"] = (offer.wo * offer.duration).sum()
stepO["po"] = stepO["ro"] / stepO["qo"]
print(
    "'Expected' situation at offer:\n"
    + f". Offtake: {stepO['qo']:,.0f} MWh\n"
    + f"           {stepO['po']:.2f} Eur/MWh\n"
    + f". Sourced: {stepO['qs']:,.0f} MWh\n"
    + f"           {stepO['ps']:.2f} Eur/MWh\n"
    + ". Market prices: {0[p_peak]:.2f} (peak), {0[p_offpeak]:.2f} (offpeak), {0[p_base]:.2f} (base)".format(
        lb.prices.convert.tseries2bpoframe(offer.pu, "AS").iloc[0, :]
    )
)

# %% SIMULATIONS (FINAL SITUATION)

sims = []
for i in range(10_000):
    # situation at end of long-term step (L)
    sim = pd.DataFrame({"pu": p_sim(), "wo": wo_sim()}).dropna()
    sim["wu"] = sim.wo - offer.ws  # open volume at that time
    sim["ru"] = sim.wu * sim.duration * sim.pu  # value of that volume
    stepL = pd.Series(dtype=float)
    stepL["ro"] = sim.ru.sum() + offer.rs.sum()
    stepL["qo"] = (sim.wo * sim.duration).sum()
    stepL["po"] = stepL["ro"] / stepL["qo"]
    stepL["delta_po"] = stepL["po"] - stepO["po"]
    stepL["delta_rmp"] = -stepL["qo"] * stepL["delta_po"]
    sims.append(stepL)
    if i % 100 == 0:
        print(f"{i}...")
print("ready.")
sims = pd.DataFrame(sims)


# %% ANALYSIS

source_vals = sims.delta_po
loc, scale = norm.fit(source_vals)
x = source_vals.sort_values().to_list()
x_fit = np.linspace(1.3 * min(x) - 0.3 * max(x), -0.3 * min(x) + 1.3 * max(x), 300)
y_fit = norm(loc, scale).cdf(x_fit)
fig, ax = plt.subplots(1, 1)
ax.title.set_text("Churn risk")
ax.xaxis.label.set_text("Eur/MWh")
ax.yaxis.label.set_text("cumulative fraction")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.hist(
    x,
    cumulative=True,
    color="orange",
    density=True,
    bins=np.array((*x, x[-1] + 0.1)),
    histtype="step",
    linewidth=1,
)
ax.plot(
    x_fit,
    y_fit,
    color="k",
    linewidth=1,
    label=f"Fit:\nmean: {loc:.2f}, std: {scale:.2f}",
)
ax.legend()

# We want to cover a certain quantile.
quantile = 0.8
p_premium = norm(loc, scale).ppf(quantile)
ax.text(
    loc + scale,
    0.4,
    f"if quantile = {quantile:.0%}, then\n   -> premium = {p_premium:.2f} Eur/MWh",
)

