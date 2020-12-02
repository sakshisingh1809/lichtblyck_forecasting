"""
Module to calculate the Beschaffungskostenaufschlag.

2020_10 RW
"""

import lichtblyck as lb
from lichtblyck import tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

# %% OFFTAKE

# Get prognosis, prepare dataframe.
prog = pd.read_excel(
    "scripts/2020_10 Beschaffungskomponente Power B2C/20201005_Input für die Beschaffungskomponente.xlsx",
    header=0,
    skiprows=1,
    index_col=0,
    usecols=[0, 1, 2, 3, 4],
    names=["ts_local_right", "w_exp", "w_100pct", "w_certain", "p_pfc"],
)
prog = lb.tools.set_ts_index(prog)
p_pfc = prog.pop("p_pfc")

# Get other prognosis curves.
def w_offtake(churn_2021_as_frac_of_100pct: float):
    """Function to calculate prognosis paths for arbitrary churn fractions."""
    churnpath_of_expvolume = 1 - prog["w_exp"] / prog["w_100pct"]
    churnfrac_of_expvolume = 1 - prog["w_exp"].mean() / prog["w_100pct"].mean()
    churnpath = (
        churnpath_of_expvolume * churn_2021_as_frac_of_100pct / churnfrac_of_expvolume
    )
    return prog["w_100pct"] * (1 - churnpath)


worst, best = 0.12, 0.05  # churn fractions, yearly aggregates

# Simulate actual offtake path, between lower and upper bound (uniform distribution).
def w_sim():
    churn2021 = np.random.uniform(best, worst)
    return w_offtake(churn2021).rename("w")


# Quick visual check.
fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
ref = prog["w_100pct"].resample("D").mean()
for _ in range(300):
    w = w_sim().resample("D").mean()
    ax[0].plot(w, alpha=0.03, color="k")
    ax[1].plot(w / ref, alpha=0.03, color="k")
ax[0].plot(w, color="k")  # accent on one
ax[1].plot(w / ref, color="k")  # accent on one
ax[0].plot(prog[["w_100pct", "w_exp", "w_certain"]].resample("D").mean(), color="r")
ax[1].plot(
    prog[["w_100pct", "w_exp", "w_certain"]].resample("D").mean().div(ref, axis=0),
    color="r",
)


# %% PRICES

# Get prices with correct expectation value (=average).
p_sims = pd.read_csv(
    "scripts/2020_10 Beschaffungskomponente Ludwig/MC_NL_HOURLY_PWR.csv"
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
p_sims = tools.set_ts_index(p_sims)
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
for _ in range(300):
    ax.plot(p_sim().resample("D").mean(), alpha=0.03, color="k")
ax.plot(p_sim().resample("D").mean(), color="k")  # accent on one
ax.plot(p_pfc.resample("D").mean(), color="r")


# %% EXPECTATION (INITIAL SITUATION)

# Do value hedge to find out, what futures volumes are bought before the start of the year.
# Option 1: futures procurement at level of certain volume (prog['w_certain'])
# Option 2: futures procurement at level of expected volume (prog['w_exp'])
w_hedge = lb.prices.w_hedge_long(prog["w_certain"], p_pfc, "YS")
p_hedge = tools.wavg(p_pfc, w_hedge)  # 46.80, 44.93

initial = pd.DataFrame(columns=[[], []], index=w_hedge.index)  # 2-level columns
initial[("offtake", "w")] = prog.w_exp
initial[("hedge", "w")] = w_hedge
initial[("hedge", "p")] = p_pfc * 44.93 / p_hedge
initial[("open", "w")] = initial.offtake.w - initial.hedge.w
initial[("open", "p")] = p_pfc
r_initial = initial.open.r.sum() + initial.hedge.r.sum()
q_initial = initial.offtake.q.sum()
p_initial = r_initial / q_initial


# %% SIMULATIONS (FINAL SITUATION)

sims = []
for _ in range(10_000):
    sim = pd.DataFrame({("spot", "p"): p_sim(), ("offtake", "w"): w_sim()}).dropna()
    sim[("spot", "w")] = sim.offtake.w - initial.hedge.w
    sim[("spot", "r")] = sim.spot.q * sim.spot.p
    r_final = sim.spot.r.sum() + initial.hedge.r.sum()
    q_final = sim.offtake.q.sum()
    p_final = r_final / q_final

    p_premium = p_final - p_initial
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
    if _ % 1000 == 0:
        print(f"{_}...")
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
quantile = 0.95
r_premium = norm(loc, scale).ppf(quantile)
p_premium = r_premium / q_initial
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
