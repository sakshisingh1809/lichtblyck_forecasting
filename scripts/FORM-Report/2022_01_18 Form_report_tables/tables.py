# %% IMPORTS AND PREP

import lichtblyck as lb
import pandas as pd
import datetime as dt
import math
from scipy import stats

lb.belvis.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")

vola = 1.2  # /calyear


def var(pfs: lb.PfState) -> pd.DataFrame:
    """Calculate value at risk due to open positions."""
    # Current state.
    volume = pfs.unsourced.q
    price = pfs.unsourced.p
    price[price.isna()] = 0
    # Possible changes.
    caldays = 4
    calyears = caldays / 365.24
    # Distribution.
    sigma = vola * math.sqrt(calyears)  # as fraction
    mu = -0.5 * sigma ** 2  # as fraction
    series = {}
    for label, quantile in {"fall": 0.05, "rise": 0.95}.items():
        exponent = stats.norm(mu, sigma).ppf(quantile)
        multiplication_factor = math.exp(exponent)
        pfs_q = pfs.set_unsourcedprice(pfs.unsourcedprice * multiplication_factor)
        series[f"{label}_delta_r"] = pfs_q.pnl_cost.r - pfs.pnl_cost.r
        series[f"{label}_delta_p"] = pfs_q.pnl_cost.p - pfs.pnl_cost.p
    return pd.DataFrame(series)


def big_df(pfs, freq) -> pd.DataFrame:
    """Create big dataframe with selected information."""
    aggpfs = pfs.asfreq(freq)
    dfs = {}
    dfs["offtake"] = -1 * aggpfs.offtake.df("q")
    dfs["hedged"] = pd.DataFrame(
        {"fraction": aggpfs.sourcedfraction, "p": aggpfs.sourced.p}
    )
    if pfs.index.freq in ["H", "15T"]:
        market = pfs.unsourcedprice.po(freq)
        dfs["current_market_offpeak"] = market["offpeak"][["p"]]
        dfs["current_market_peak"] = market["peak"][["p"]]
    else:
        market = pfs.unsourcedprice.p.resample(freq).mean()
        dfs["current_market"] = pd.DataFrame({"p": market})
    dfs["portfolioprice"] = aggpfs.pnl_cost.df("p")
    dfs["var"] = var(aggpfs)
    return pd.concat(dfs, axis=1)


def write_to_excel(commodity, pfname, thisyear: bool = True):
    if thisyear:
        start, end, freq = "2022", "2023", "MS"
    else:
        start, end, freq = "2022", "2025", "AS"
    pfs = lb.portfolios.pfstate(commodity, pfname, start, end)

    df = big_df(pfs, freq).tz_localize(None)
    if freq == "MS":  # add sum.
        df2 = big_df(pfs, "AS")
        df2.index = ["Yearly Agg."]
        df = pd.concat([df, df2], axis=0)
    df.pint.dequantify().to_excel(writer, sheet_name=f"{commodity}_{pfname}")


# %% GET DATA AND WRITE TO EXCEL

writer = pd.ExcelWriter(f"state_on_{dt.date.today()}.xlsx", engine="xlsxwriter")

write_to_excel("power", "B2C_HH_LEGACY", True)
write_to_excel("power", "B2C_P2H_LEGACY", True)
write_to_excel("power", "B2C_HH_NEW", False)
write_to_excel("power", "B2C_P2H_NEW", False)
write_to_excel("gas", "B2C_LEGACY", True)
write_to_excel("gas", "B2C_NEW", False)
write_to_excel("gas", "B2B", False)


#%%

writer.save()
writer.close()  # will give warning ('already closed') but still necessary to open in excel
