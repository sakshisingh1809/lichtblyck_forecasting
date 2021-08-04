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

# Get current situation, prepare dataframe.
current = pd.read_excel(
    Path(__file__).parent / "20210804_090841_Zeitreihenbericht.xlsx",
    header=1,
    index_col=0,
    usecols=[0, 1, 4, 6, 8],
    names=["ts_local_right", "pu", "wo", "ws", "rs"],
)
current = lb.set_ts_index(current, bound="right")
current.wo = -current.wo
# Correction: the sourced volume is actually for this PF but also for WP and NSP.
contrib = 0.98
current.ws *= contrib
current.rs *= contrib
current['ps'] = current.rs / (current.ws * current.duration)
current["wu"] = current.wo - current.ws
current["ro"] = current.rs + current.wu * current.duration * current.pu



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
p_sims = pd.read_csv(Path(__file__).parent     / "MC_NL_HOURLY_PWR.csv")
# . Clean: make timezone-aware
p_sims = p_sims.set_index("deldatetime")
p_sims.index = pd.to_datetime(p_sims.index, format='%d-%m-%Y %H:%M')
p_sims.index.freq = p_sims.index.inferred_freq
# . . The goal: timezone-aware index.
idx = p_sims.index # the original: 24 hours on each day. This is incorrect.
idx_EuropeBerlin = pd.date_range(idx[0], idx[-1], freq=idx.freq, tz="Europe/Berlin")
# . . The corresponding local datetime.
idx_local = idx_EuropeBerlin.tz_localize(None)
# . . Use local datetime to find correct values for the timezone-aware datetime.
p_sims = p_sims.loc[idx_local, :].set_index(idx_EuropeBerlin)
# . . Rename index
p_sims = lb.set_ts_index(p_sims)

# . Clean: make arbitrage free
p_sims = lb.changefreq_avg(p_sims, current.pu.index.freq) # correct frequency 
factor = (current.pu / p_sims.mean(axis=1)).dropna() # find correction factor
p_sims = p_sims.multiply(factor, axis=0).dropna() # apply

# Get a price simulation.
def p_sim():
    i = np.random.randint(len(p_sims.columns))
    return p_sims.iloc[:, i]


# Quick visual check.
fig, ax = plt.subplots(figsize=(16, 10))
for sim in range(300):
    ax.plot(lb.changefreq_avg(p_sim(), "D"), alpha=0.03, color="k")
ax.plot(lb.changefreq_avg(p_sim(), "D"), color="k")  # accent on one
ax.plot(lb.changefreq_avg(current.pu, 'D'), color="r")


# %% OFFER (=INITIAL SITUATION)

# Find 'eventual' sourced volume at offer time, and current (best-guess) offer price.
w_hedge_missing = lb.hedge(current.wu, current.pu, "AS", 'val', bpo=True)
w_hedge = current.ws + w_hedge_missing
r_hedge = current.rs + w_hedge_missing * w_hedge_missing.duration * current.pu
p_hedge = r_hedge / (w_hedge * w_hedge.duration)
ro = r_hedge + (current.wo - w_hedge) * current.duration * current.pu
po = ro / (current.wo * current.duration)


offer = current.copy()
offer['ws'] = w_hedge
offer['rs'] = r_hedge
offer['ps'] = p_hedge
offer['wu'] = offer.wo - offer.ws
offer['ro'] = ro

#%%%%%%









offer[("offtake", "w")] = current.wo
offer[("hedge", "w")] = w_hedge
offer[("hedge", "p")] = current.pu * 44.93 / p_hedge
offer[("open", "w")] = offer.offtake.w - offer.hedge.w
offer[("open", "p")] = p_pfc
offer[("open", "r")] = offer.open.w * offer.duration * offer.open.p
offer[("hedge", "r")] = offer.hedge.w * offer.duration * offer.hedge.p
ro_offer = offer.open.r.sum() + offer.hedge.r.sum()
qo_offer = (offer.offtake.w * offer.duration).sum()
po_offer = ro_offer / qo_offer


# %% SIMULATIONS (FINAL SITUATION)

sims = []
for i in range(1_000):
    sim = pd.DataFrame({("spot", "p"): p_sim(), ("offtake", "w"): wo_sim()}).dropna()
    sim[("spot", "w")] = sim.offtake.w - offer.hedge.w
    sim[("spot", "r")] = sim.spot.w * sim.duration * sim.spot.p
    r_final = sim.spot.r.sum() + offer.hedge.r.sum()
    q_final = (sim.offtake.w * sim.duration).sum()
    p_final = r_final / q_final

    p_premium = p_final - po_offer
    r_premium = p_premium * q_final
    # r_premium2 = initial.hedge.r.sum() + sim.spot.r.sum() - p_initial * q_final
    # assert (np.isclose(r_premium, r_premium2)) #check; should be same value
    sims.append(
        {
            ("final", "q"): q_final,
            ("final", "p"): p_final,
            ("premium", "p"): p_premium,
            ("premium", "r"): r_premium,
        }
    )
    if i % 10 == 0:
        print(f"{i}...")
print("ready.")
sims = pd.DataFrame(sims)
sims.columns = pd.MultiIndex.from_tuples(sims.columns)


# %% ANALYSIS

source_vals = sims.premium.r
loc, scale = norm.fit(source_vals)
x = source_vals.sort_values().unique()
x_fit = np.linspace(1.3 * min(x) - 0.3 * max(x), -0.3 * min(x) + 1.3 * max(x), 300)
y_fit = norm(loc, scale).cdf(x_fit)
fig, ax = plt.subplots(1, 1)
ax.title.set_text("Beschaffungskostenaufschlag")
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
r_premium = norm(loc, scale).ppf(quantile)
p_premium = r_premium / qo_offer
r_etl = lb.analyses.expected_shortfall(loc, scale, quantile=quantile)
ax.text(
    loc + scale,
    0.4,
    f"if quantile = {quantile:.0%}, then\n"
    + f"   -> premium = {p_premium:.2f} Eur/MWh\n"
    + "   -> expected shortfall = "
    + f"{r_etl:,.0f}".replace(",", " ")
    + " Eur",
)
pre_exp = np.array((p_premium, r_premium, r_etl))

# %%
