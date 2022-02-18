"""Calculate the long-term temprisk from current prices and reference values. (per month)"""

from typing import Callable

import numpy as np
import lichtblyck as lb
import pandas as pd
from tqdm.auto import tqdm
from scipy import stats

lb.belvis.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")
pfname = "B2C_P2H_LEGACY"
ref = lb.portfolios.pfstate("power", pfname, "2022", "2023")


# Current situation based on reference prices.


ref_prices = {
    "B2C_P2H_LEGACY": [
        76.50,
        70.16,
        54.40,
        57.18,
        38.83,
        13.38,
        -25.20,
        -37.26,
        17.43,
        49.60,
        57.93,
        59.25,
    ],
}
ref_prices = {
    name: lb.PfLine(
        {
            "p": pd.Series(
                vals,
                pd.date_range("2022", freq="MS", periods=12, tz="Europe/Berlin"),
            )
        }
    )
    for name, vals in ref_prices.items()
}


m_ref = -ref.changefreq("MS").offtake.volume * ref_prices[pfname]
q_ref = m_ref.changefreq("QS")
a_ref = m_ref.changefreq("AS")


# %% Possible future situations: different offtakes (temperature-related) and different prices.

current_prices = ref.unsourcedprice


def pricecurve_sim(
    current_prices: lb.PfLine, vola_peak: float, vola_offpeak: float
) -> Callable:
    """Returns function to calculate price curve at certain quantile. vola_peak and vola_offpeak
    are given as fraction_per_calenderyear."""

    f_peak = lambda t, q: lb.analyse.multiplication_factor(vola_peak, t, q)
    f_offpeak = lambda t, q: lb.analyse.multiplication_factor(vola_offpeak, t, q)
    now = pd.Timestamp.now(tz="Europe/Berlin")

    def pcpart_sim(start, p):
        # returns function to calculate price curve *of a single month* at certain quantile.
        t = (start - now).total_seconds() / 3600 / 24 / 365
        mask = lb.is_peak_hour(p.index)
        if t < 0:  # month start in the past
            return lambda q: p
        else:
            return lambda q: p * (mask * f_peak(t, q) + (~mask) * f_offpeak(t, q))

    pcparts = [pcpart_sim(start, p) for start, p in current_prices.p.resample("MS")]

    def pricecurve(q: float) -> lb.PfLine:
        return lb.PfLine({"p": pd.concat([pcpart(q) for pcpart in pcparts], axis=0)})

    return pricecurve


current_offtake = ref.offtake.volume


def offtakecurve_sim(current_offtake: lb.PfLine, scale_factors: float) -> Callable:
    """Returns function to calculate the offtake curve at certain quantile.
    Factor to multiply current monthly offtake with is sampled from normal distribution
    with mu = 1 and scale_factors as sigma (12 values)."""

    # to improve: currently no correlation between months
    scale_factors[0] = 1e-5  # no change in Jan; almost over.
    normfn = [stats.norm(1, sf) for sf in scale_factors]

    def offtakecurve() -> lb.PfLine:

        factors = lb.changefreq_avg(
            pd.Series(
                [nfn.ppf(np.random.rand()) for nfn in normfn],
                pd.date_range("2022", freq="MS", periods=12, tz="Europe/Berlin"),
            ),
            "15T",
        )
        return current_offtake * factors

    return offtakecurve


# %% do simulations.
scale_factors = [
    0.143429987,
    0.148861823,
    0.163135795,
    0.203908135,
    0.34076071,
    0.527289571,
    0.219551815,
    0.259229265,
    0.413602577,
    0.239499397,
    0.101652805,
    0.14890842,
]

pc = pricecurve_sim(current_prices, 1, 1)
oc = offtakecurve_sim(current_offtake, scale_factors)


m_sims = {}  # total cost for each month and simulation
q_sims = {}
a_sims = {}
n = 30  # datapoints per dimension
for quantile_pc in tqdm(np.linspace(0.5 / n, 1 - 0.5 / n, n)):
    price = pc(quantile_pc)
    pfs_simp = ref.set_unsourcedprice(price)
    for j in tqdm(range(15)):
        offtake = oc()
        m_sim = pfs_simp.set_offtakevolume(offtake).changefreq("MS").pnl_cost
        q_sim = m_sim.changefreq("QS")
        a_sim = q_sim.changefreq("AS")
        m_sims[(quantile_pc, j)] = m_sim.r
        q_sims[(quantile_pc, j)] = q_sim.r
        a_sims[(quantile_pc, j)] = a_sim.r

# wide dataframes with a column for each simulation.
df_m_sims = pd.concat(m_sims, axis=1)
df_q_sims = pd.concat(q_sims, axis=1)
df_a_sims = pd.concat(a_sims, axis=1)
df_m_premiums = df_m_sims.sub(m_ref.r, axis=0).div(m_ref.q, axis=0)
df_q_premiums = df_q_sims.sub(q_ref.r, axis=0).div(q_ref.q, axis=0)
df_a_premiums = df_a_sims.sub(a_ref.r, axis=0).div(a_ref.q, axis=0)

# %% VISUALIZE DISTRIBUTIONS.

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm


def visualze_fn(monthly_premiums_per_simulation, quartely_premiums_per_simulation):
    def fn(show: str = "M"):
        if show == "M":
            shape = (3, 4)
            df_premiums = monthly_premiums_per_simulation
        elif show == "Q":
            shape = (2, 2)
            df_premiums = quartely_premiums_per_simulation

        fig, axes = plt.subplots(*shape, sharey=True, figsize=(16, 10))

        for ts, ax in zip(df_premiums.index, axes.flatten()):
            #  Distribution.
            source_vals = df_premiums.loc[ts, :].T.pint.m
            loc, scale = norm.fit(source_vals)
            x = source_vals.sort_values().tolist()
            x_fit = np.linspace(
                1.1 * min(x) - 0.1 * max(x), -0.1 * min(x) + 1.1 * max(x), 300
            )
            y_fit = norm(loc, scale).cdf(x_fit)
            ax.title.set_text(f"{ts}.")
            ax.xaxis.label.set_text("Eur/MWh")
            ax.yaxis.label.set_text("cumulative fraction")
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
            ax.hist(
                x,
                cumulative=True,
                color="orange",
                density=True,
                bins=x + [x[-1] + 0.1],
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

    return fn


viz = visualze_fn(df_m_premiums, df_q_premiums)

viz("M")
viz("Q")


# %% CALCULATE PREMIUMS.

quantiles = [0.5, 0, 58, 0.8, 0.9, 0.95]
df_m_premium = pd.concat([df_m_premiums.quantile(q, axis=1) for q in quantiles], axis=1)
df_q_premium = pd.concat([df_q_premiums.quantile(q, axis=1) for q in quantiles], axis=1)
df_a_premium = pd.concat([df_a_premiums.quantile(q, axis=1) for q in quantiles], axis=1)
df_m_premium.tz_localize(None).pint.dequantify().to_excel("lttr_monthly.xlsx")
df_q_premium.tz_localize(None).pint.dequantify().to_excel("lttr_quarterly.xlsx")

# %%
