"""
Module to calculate the Beschaffungskostenaufschlag.

Abbreviations:
pu = market price [Eur/MWh]
ps = sourced price [Eur/MWh]
rs = sourced cost [EUR]
ws = sourced power [MW]
qs = sourced volume [MWh]
wo = offtake power [MW]
qo = offtake volume [MWh]
"""

# %% IMPORTS

import lichtblyck as lb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import functools
from scipy.stats import norm
from pathlib import Path
from tqdm import tqdm

# %% PREPARATIONS

pf = "P2H"  # "P2H" or "B2C"

if pf == "B2C":
    usecols = [0, 1, 3, 4, 5, 6]  # B2C
else:
    usecols = [0, 1, 7, 8, 9, 10]  # P2H

# Get current situation, prepare dataframe.
current = pd.read_excel(
    Path(".").parent / "Copy of Beschaffungskomponente_100%.xlsx",
    sheet_name="Sheet1",
    header=2,
    index_col=0,
    usecols=usecols,
    names=["ts_local_right", "pu", "wo100", "wo", "ws", "rs"],
)
current = lb.set_ts_index(current, bound="right")

# Calculate additional information.
current["ps"] = current.rs / (current.ws * current.duration)
current["wu"] = current.wo - current.ws
current["ru"] = current.wu * current.duration * current.pu
current["ro"] = current.rs + current.ru


#%% OFFTAKE

# Get other offtake curves.
def wo(churn_2021_as_frac_of_100pct: float):
    """Function to calculate prognosis paths for arbitrary churn fractions."""
    churnpath_of_certainvolume = 1 - current.wo / current.wo100
    churnfrac_of_certainvolume = 1 - current.wo.mean() / current.wo100.mean()
    churnpath = (
        churnpath_of_certainvolume
        * churn_2021_as_frac_of_100pct
        / churnfrac_of_certainvolume
    )
    return current.wo100 * (1 - churnpath)


# Scenarios between which to simulate.
# churn fractions, yearly aggregates
expected = 1 - current.wo.mean() / current.wo100.mean()
best, worst = 0, expected * 2
current["wo_low"] = wo(worst)
current["wo_exp"] = wo(expected)
current["wo_upp"] = wo(best)

# Simulate actual offtake path, between lower and upper bound (uniform distribution).
def wo_sim():
    churn2022 = np.random.uniform(best, worst)
    return wo(churn2022).rename("w")


# Quick visual check.
fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
ref = lb.changefreq_avg(current.wo100, "D")
for sim in tqdm(range(100)):
    w = lb.changefreq_avg(wo_sim(), "D")
    ax[0].plot(w, alpha=0.06, color="k")
    ax[1].plot(w / ref, alpha=0.06, color="k")
ax[0].plot(lb.changefreq_avg(current.wo, "D"), color="r")
ax[1].plot(
    lb.changefreq_avg(current.wo, "D").div(ref, axis=0), color="r",
)
ax[0].plot(lb.changefreq_avg(current.wo100, "D"), color="b")
ax[1].plot(
    lb.changefreq_avg(current.wo100, "D").div(ref, axis=0), color="b",
)
ax[0].yaxis.label.set_text("MW")
ax[1].yaxis.label.set_text("as fraction of 'current' offtake")
ax[1].yaxis.set_major_formatter("{:.0%}".format)


# %% PRICES

# Get prices with correct expectation value (=average).
p_sims = pd.read_csv(Path(".").parent / "MC_NL_HOURLY_PWR.csv")
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
for sim in tqdm(range(100)):
    ax.plot(lb.changefreq_avg(p_sim(), "D"), alpha=0.06, color="k")
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
for i in tqdm(range(1000)):
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


# %%
