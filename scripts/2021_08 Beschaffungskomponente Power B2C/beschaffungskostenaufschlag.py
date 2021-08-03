"""
Module to calculate the Beschaffungskostenaufschlag.
"""

# %% IMPORTS

import lichtblyck as lb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
from pathlib import Path

# %% OFFTAKE

# Get prognosis, prepare dataframe.
prog = pd.read_excel(
    Path(__file__).parent / "20210728_144427_Zeitreihenbericht.xlsx",
    header=1,
    index_col=0,
    usecols=[0, 1, 7],
    names=["ts_local_right", "p_pfc", "w_exp"],
)
prog = lb.set_ts_index(prog, bound="right")
p_pfc = prog.pop("p_pfc")
prog = -prog

# Get other prognosis curves.
def w_offtake(churnstart: float = 0, churnend: float = 0.1):
    """Function to calculate prognosis paths.
    churnstart: (excess) churn at start of year, i.e., compared to expected path.
    churnend: (excess) churn at end of year, i.e., compared to expected path."""
    yearfrac = (prog.index - prog.index[0]) / (prog.index[-1] - prog.index[0])
    churnpath = churnstart * (1 - yearfrac) + churnend * yearfrac
    return prog["w_exp"] * (1 - churnpath)


# Scenarios between which to simulate.
worst = {"churnstart": 0.02, "churnend": 0.1}
best = {"churnstart": -0.02, "churnend": -0.1}


# Simulate actual offtake path, between lower and upper bound (uniform distribution).
def w_sim():
    howbad = np.random.uniform()
    start = howbad * worst["churnstart"] + (1 - howbad) * best["churnstart"]
    end = howbad * worst["churnend"] + (1 - howbad) * best["churnend"]
    return w_offtake(start, end).rename("w")


# Quick visual check.
fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
ref = prog["w_exp"].resample("D").mean()
for sim in range(300):
    w = w_sim().resample("D").mean()
    ax[0].plot(w, alpha=0.03, color="k")
    ax[1].plot(w / ref, alpha=0.03, color="k")
ax[0].plot(w, color="k")  # accent on one
ax[1].plot(w / ref, color="k")  # accent on one
ax[0].plot(prog[["w_exp"]].resample("D").mean(), color="r")
ax[1].plot(
    prog[["w_exp"]].resample("D").mean().div(ref, axis=0), color="r",
)
ax[0].yaxis.label.set_text('MW')
ax[1].yaxis.label.set_text('as fraction of expected offtake')
ax[1].yaxis.set_major_formatter("{:.0%}".format)


# %% PRICES

# Get prices with correct expectation value (=average).
p_sims = pd.read_csv(
    Path(__file__).parent.parent
    / "2020_10 Beschaffungskomponente Ludwig/MC_NL_HOURLY_PWR.csv"
)
# . Clean: make timezone-aware
p_sims = p_sims.set_index("deldatetime")
p_sims.index = pd.DatetimeIndex(p_sims.index)
p_sims.index.freq = p_sims.index.inferred_freq
idx = p_sims.index  # original: 24 hours on each day. Not correct
idx_EuropeBerlin = pd.date_range(
    idx[0], idx[-1], freq=idx.freq, tz="Europe/Berlin"
)  # what we want
idx_local = idx_EuropeBerlin.tz_localize(None)  # corresponding local time
p_sims = p_sims.loc[idx_local, :].set_index(
    idx_EuropeBerlin
)  # Final dataframe with correct timestamps
# . Rename index
p_sims = lb.set_ts_index(p_sims)
# . Make arbitrage-free to pfc
# factor = (p_pfc / p_sims.mean(axis=1)).dropna()
# p_sims = p_sims.multiply(factor, axis=0).dropna()
factor = (p_pfc / p_sims.mean(axis=1).resample(p_pfc.index.freq).ffill()).dropna()
p_sims = p_sims.resample(p_pfc.index.freq).ffill().multiply(factor, axis=0).dropna()

# Get a price simulation.
def p_sim():
    i = np.random.randint(len(p_sims.columns))
    return p_sims.iloc[:, i]


# Quick visual check.
fig, ax = plt.subplots(figsize=(16, 10))
for sim in range(300):
    ax.plot(p_sim().resample("D").mean(), alpha=0.03, color="k")
ax.plot(p_sim().resample("D").mean(), color="k")  # accent on one
ax.plot(p_pfc.resample("D").mean(), color="r")


# %% OFFER (=INITIAL SITUATION)

# Do value hedge to find out, what futures volumes are bought before the start of the year.
# Forward procurement at level of expected volume (prog['w_exp'])
w_hedge = lb.hedge(prog["w_exp"], p_pfc, "AS")
p_hedge = lb.tools.wavg(p_pfc, w_hedge)  # 46.80, 44.93

offer = pd.DataFrame(columns=[[], []], index=w_hedge.index)  # 2-level columns
offer[("offtake", "w")] = prog.w_exp
offer[("hedge", "w")] = w_hedge
offer[("hedge", "p")] = p_pfc * 44.93 / p_hedge
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
    sim = pd.DataFrame({("spot", "p"): p_sim(), ("offtake", "w"): w_sim()}).dropna()
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
