#%%
import lichtblyck as lb
import pandas as pd
import datetime as dt

lb.belvis.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")


# %%


def big_df(aggpfs, income_prices):
    """Create big dataframe with selected information."""
    dfs = {
        "offtake": pd.DataFrame({"q": -1 * aggpfs.offtake.q, "p": income_prices}),
        "sourced": pd.DataFrame(
            {
                "fraction": aggpfs.sourcedfraction,
                "q": aggpfs.sourced.q,
                "p": aggpfs.sourced.p,
            }
        ),
        "portfolioprice": aggpfs.pnl_cost.df("p"),
    }
    df = pd.concat(dfs, axis=1)
    df[("result", "r")] = df.offtake.q * (df.offtake.p - df.portfolioprice.p)
    df[("result", "p")] = df.result.r / df.offtake.q
    return df


def write_to_excel(sheetname, pfs, income_prices, freq="MS"):
    pfs = pfs.asfreq("MS")
    offtakewithprices = pfs.offtake.set_p(income_prices)
    df = big_df(
        pfs.asfreq(freq),
        offtakewithprices.asfreq(freq).p,
    ).tz_localize(None)
    # add sum.
    df2 = big_df(
        pfs.asfreq("AS"),
        offtakewithprices.asfreq("AS").p,
    )
    df2.index = ["Yearly Agg."]
    df = pd.concat([df, df2], axis=0)
    df.pint.dequantify().to_excel(writer, sheet_name=sheetname)


# %% POWER

pfname = "B2C_P2H_LEGACY"
pfs_pow = lb.portfolios.pfstate("power", pfname, "2022")
pfs = pfs_pow
income = pd.Series(
    [
        92.93,
        86.60,
        70.83,
        73.62,
        55.27,
        29.82,
        -8.75,
        -20.82,
        33.88,
        66.04,
        74.37,
        75.70,
    ],
    pd.date_range("2022", freq="MS", periods=12, tz="Europe/Berlin"),
)
income = income - 0.81 - 0.39

writer = pd.ExcelWriter(
    f"state_on_{dt.date.today()}_power_{pfname}.xlsx", engine="xlsxwriter"
)

write_to_excel("current", pfs, income)
write_to_excel("1.5p", pfs.set_unsourcedprice(pfs.unsourcedprice * 1.5), income)
write_to_excel("0.5p", pfs.set_unsourcedprice(pfs.unsourcedprice * 0.5), income)

writer.save()
writer.close()  # will give warning ('already closed') but still necessary to open in excel


# %% GAS

pfname = "B2C_LEGACY"
pfs_gas = lb.portfolios.pfstate("gas", pfname, "2022")
pfs = pfs_gas
income = pd.Series(
    [
        23.045,
        22.121,
        17.768,
        28.195,
        27.228,
        21.752,
        19.025,
        19.725,
        22.698,
        39.974,
        45.369,
        48.523,
    ],
    pd.date_range("2022", freq="MS", periods=12, tz="Europe/Berlin"),
)

writer = pd.ExcelWriter(
    f"state_on_{dt.date.today()}_gas_{pfname}.xlsx", engine="xlsxwriter"
)

write_to_excel("current", pfs, income)
write_to_excel("1.5p", pfs.set_unsourcedprice(pfs.unsourcedprice * 1.5), income)
write_to_excel("0.5p", pfs.set_unsourcedprice(pfs.unsourcedprice * 0.5), income)

writer.save()
writer.close()  # will give warning ('already closed') but still necessary to open in excel

# %%
