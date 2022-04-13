"""Calculate churnrisk, with current portfolio prices as reference."""

#%%
from dataclasses import dataclass
from typing import Callable
import lichtblyck as lb
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import winsound


lb.belvis.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")


pfname = "B2C_P2H_LEGACY"
ref = lb.portfolios.pfstate("power", pfname, "2022", "2023")


# %% Dataclass


@dataclass
class SimResult:
    quant_pc: float
    quant_oc: float
    pfs: lb.PfState
    O_qo: pd.Series
    O_po: pd.Series
    L_qo: pd.Series
    L_po: pd.Series
    L_delta_qo: pd.Series
    L_delta_po: pd.Series


# %% Current situation based on current volumes and prices.

m_ref = ref.asfreq("MS").pnl_cost
q_ref = m_ref.asfreq("QS")
a_ref = m_ref.asfreq("AS")


# %% Possible future situations: different offtakes (churn uncertainty) and different prices.

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


def offtakecurve_sim(
    current_offtake: lb.PfLine, uncertainty_per_year: float
) -> Callable:
    """Returns function to calculate the offtake curve at certain quantile.
    uncertainty_per_year = 0.12 means that offtake that is to be delivered in one year
    is known certain to within +-12%; inside this uncertainty the volume is distributed
    uniformly."""

    now = pd.Timestamp.now(tz="Europe/Berlin")
    i = current_offtake.index
    t = pd.Series((i - now).total_seconds() / 3600 / 24 / 365, i)
    t[t < 0] = 0
    uncertainty_curve = t * uncertainty_per_year

    def offtakecurve(q: float) -> lb.PfLine:
        factor = 1 + uncertainty_curve * (2 * q - 1)
        return current_offtake * factor

    return offtakecurve


# %% do simulations.
pc = pricecurve_sim(current_prices, 1.4, 1.4)
oc = offtakecurve_sim(current_offtake, 0.08)

m_sims = {}  # total cost for each month and simulation
q_sims = {}
a_sims = {}
n = 15  # datapoints per dimension
for quantile_pc in tqdm(np.linspace(0.5 / n, 1 - 0.5 / n, n)):
    price = pc(quantile_pc)
    pfs_simp = ref.set_unsourcedprice(price)
    for quantile_oc in tqdm(np.linspace(0.5 / n, 1 - 0.5 / n, n)):
        offtake = oc(quantile_oc)
        m_sim = pfs_simp.set_offtakevolume(offtake).asfreq("MS")
        q_sim = m_sim.asfreq("QS")
        a_sim = q_sim.asfreq("AS")
        for container, pfs, ref_pfl in zip(
            [m_sims, q_sims, a_sims], [m_sim, q_sim, a_sim], [m_ref, q_ref, a_ref]
        ):
            container[(quantile_pc, quantile_oc)] = SimResult(
                quantile_pc,
                quantile_oc,
                pfs,
                ref_pfl.q,
                ref_pfl.p,
                pfs.pnl_cost.q,
                pfs.pnl_cost.p,
                pfs.pnl_cost.q - ref_pfl.q,
                pfs.pnl_cost.p - ref_pfl.p,
            )


winsound.Beep(2500, 1000)

# wide dataframes with a column for each simulation with the needed premium in that simulation.
df_m_premiums = pd.concat({k: sim.L_delta_po for k, sim in m_sims.items()}, axis=1)
df_q_premiums = pd.concat({k: sim.L_delta_po for k, sim in q_sims.items()}, axis=1)
df_a_premiums = pd.concat({k: sim.L_delta_po for k, sim in a_sims.items()}, axis=1)

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
            loc, scale = norm.fit(source_vals.values)
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

        return fig

    return fn


viz = visualze_fn(df_m_premiums, df_q_premiums)
viz("M").savefig(f"{pfname}_distribution_M.png")
viz("Q").savefig(f"{pfname}_distribution_Q.png")


# %% CALCULATE QUANTILES

quantiles = [0.5, 0.58, 0.8, 0.9, 0.95]
df_m_premium = pd.concat([df_m_premiums.quantile(q, axis=1) for q in quantiles], axis=1)
df_q_premium = pd.concat([df_q_premiums.quantile(q, axis=1) for q in quantiles], axis=1)
df_a_premium = pd.concat([df_a_premiums.quantile(q, axis=1) for q in quantiles], axis=1)
df_m_premium.tz_localize(None).pint.dequantify().to_excel(
    f"churnrisk_{pfname}_monthly.xlsx"
)
df_q_premium.tz_localize(None).pint.dequantify().to_excel(
    f"churnrisk_{pfname}_quarterly.xlsx"
)

# %%
