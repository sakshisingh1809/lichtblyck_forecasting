"""Script to calculate the PNL impact of weather on April 2022."""

# %%

import lichtblyck as lb
import pandas as pd


# %%
lb.belvis.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")


# %% P2H

# Before.
before = lb.PfLine(
    pd.DataFrame(
        {"q": 123568, "p": 87.99},
        pd.date_range("2022-04", periods=1, freq="MS", tz="Europe/Berlin"),
    )
)
beforedf = before.df().pint.dequantify()

# After.
pfs = lb.portfolios.pfstate("power", "B2C_P2H_LEGACY", "2022-04", "2022-05")
after = pfs.asfreq("MS").pnl_cost
afterdf = after.df().pint.dequantify()

# Change.
changedf = afterdf - beforedf

# save to excel.

p2h = pd.concat({"before": beforedf, "after": afterdf, "change": changedf})
p2h.unstack(level=0).tz_localize(None).stack().to_excel("p2h.xlsx")


# %% GAS

# Before.
before = lb.PfLine(
    pd.DataFrame(
        {"q": 62439, "p": 25.14},
        pd.date_range("2022-04", periods=1, freq="MS", tz="Europe/Berlin"),
    )
)
beforedf = before.df().pint.dequantify()

# After.
pfs = lb.portfolios.pfstate("gas", "B2C_LEGACY", "2022-04", "2022-05")
after = pfs.asfreq("MS").pnl_cost
afterdf = after.df().pint.dequantify()

# Change.
changedf = afterdf - beforedf

# save to excel.

gas = pd.concat({"before": beforedf, "after": afterdf, "change": changedf})
gas.unstack(level=0).tz_localize(None).stack().to_excel("gas_b2c.xlsx")


# %% Teperatures

t = lb.tmpr.hist.tmpr("1998", "2021")
weights = [
    ["t_13", 36.28],
    ["t_4", 14.50],
    ["t_6", 10.37],
    ["t_3", 8.54],
    ["t_2", 5.61],
    ["t_7", 5.44],
    ["t_4", 4.08],
    ["t_5", 1.16],
    ["t_12", 1.07],
    ["t_12", 1.02],
    ["t_5", 0.71],
    ["t_5", 0.68],
    ["t_6", 0.64],
    ["t_5", 0.59],
    ["t_9", 0.57],
]


tmean = sum(t[w[0]] * w[1] for w in weights) / sum(w[1] for w in weights)
monthly = tmean.resample("MS").mean()
april = monthly[monthly.index.month == 4]
april.to_clipboard()

# %%
