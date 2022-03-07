# %% IMPORTS AND PREP

import lichtblyck as lb
import pandas as pd
import datetime as dt
import math
from scipy import stats

lb.belvis.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")


def var(pfs):
    """Calculate value at risk due to open positions."""
    # Current state.
    volume = pfs.unsourced.q
    price = pfs.unsourced.p
    price[price.isna()] = 0
    # Possible changes.
    vola = 1.4  # /calyear
    caldays = 4
    calyears = caldays / 365.24
    # Distribution.
    sigma = vola * math.sqrt(calyears)  # as fraction
    mu = -0.5 * sigma ** 2  # as fraction
    series = {}
    for label, quantile in {"fall": 0.05, "rise": 0.95}.items():
        exponent = stats.norm(mu, sigma).ppf(quantile)
        multiplication_factor = math.exp(exponent)
        change_factor = multiplication_factor - 1
        unsourced_var = lb.PfLine({"q": volume, "p": price * change_factor})
        series[label] = unsourced_var.r
    return pd.DataFrame(series)


def big_df(aggpfs, aggprices):
    """Create big dataframe with selected information."""
    dfs = {
        "offtake": -1 * aggpfs.offtake.df("q"),
        "hedged": pd.DataFrame(
            {"fraction": aggpfs.sourcedfraction, "p": aggpfs.sourced.p}
        ),
        "current_market_offpeak": aggprices["offpeak"][["p"]],
        "current_market_peak": aggprices["peak"][["p"]],
        "portfolioprice": aggpfs.pnl_cost.df("p"),
        "var": var(aggpfs),
    }
    return pd.concat(dfs, axis=1)


def write_to_excel(pfname, thisyear: bool = True):
    if thisyear:
        start, end, freq = "2022", "2023", "MS"
    else:
        start, end, freq = "2022", "2025", "AS"
    pfs = lb.portfolios.pfstate("power", pfname, start, end)

    df = big_df(pfs.asfreq(freq), pfs.unsourcedprice.po(freq)).tz_localize(None)
    if freq == "MS":  # add sum.
        df2 = big_df(pfs.asfreq("AS"), pfs.unsourcedprice.po("AS"))
        df2.index = ["Yearly Agg."]
        df = pd.concat([df, df2], axis=0)
    df.pint.dequantify().to_excel(writer, sheet_name=pfname)


# %% GET DATA AND WRITE TO EXCEL

writer = pd.ExcelWriter(f"state_on_{dt.date.today()}_test.xlsx", engine="xlsxwriter")

write_to_excel("B2C_HH_LEGACY", True)
write_to_excel("B2C_P2H_LEGACY", True)
write_to_excel("B2C_HH_NEW", False)
write_to_excel("B2C_P2H_NEW", False)

writer.save()
writer.close()  # will give warning ('already closed') but still necessary to open in excel

# %%
