"""Script to calculate the influence of weather and price changes in LUD."""

""" 
Outline.

ANY SETTING
-----------

offtake volume = f(temperature, tlp)
unsourced revenue = QHPFC * (offtake volume - sourced volume)
total costs = sourced revenue + unsourced revenue
total income = offtake volume * sales price
total pnl = total income - total costs



REFERENCE
---------

temperature = expected temperature
QHPFC = current QHPFC


SCENARIO
--------

temperature = e.g. expected temperatures + 1 degC
QHPFC = e.g. current QHPFC + 5 Eur/MWh


GRAPH
-----

x-axis: delta temperature (vs expected temperature)
y-axis: delta price (vs current QHPFC)
"""

#%%
import lichtblyck as lb
import pandas as pd
import numpy as np
import functools
from tqdm import tqdm

#%% BELVIS VALUES for calibration.

PFS = {
    "power": {
        "B2C_P2H": {
            "offtake": {
                "100%": ("LUD_NSp", "LUD_WP"),
                "certain": (
                    "LUD_NSp_SiM",
                    "LUD_WP_SiM",
                ),
            },
            "sourced": (
                "LUD_NSp",
                "LUD_WP",
            ),
        },
        "B2C_HH": {
            "offtake": ("PK_SiM", "LUD_Stg_SiM"),
            "sourced": ("PKG", "LUD_Stg"),
        },
    }
}

# Settings.
commodity = "power"
pfname = "B2C_P2H"
timerange = [
    pd.Timestamp("2021-11", tz="Europe/Berlin"),
    pd.Timestamp("2022", tz="Europe/Berlin"),
]

# Access belvis and get data.
lb.belvis.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")
offtakevolume_100 = sum(
    lb.PfLine.from_belvis_offtakevolume(commodity, belvispf, *timerange)
    for belvispf in PFS[commodity][pfname]["offtake"]["100%"]
)
offtakevolume_certain = sum(
    lb.PfLine.from_belvis_offtakevolume(commodity, belvispf, *timerange)
    for belvispf in PFS[commodity][pfname]["offtake"]["certain"]
)
sourced = sum(
    lb.PfLine.from_belvis_sourced(commodity, belvispf, *timerange)
    for belvispf in PFS[commodity][pfname]["sourced"]
)
unsourcedprice = lb.PfLine.from_belvis_forwardpricecurve(commodity, *timerange)
pfs = lb.PfState(offtakevolume_100, unsourcedprice, sourced)

#%% TEMPERATURE TO OFFTAKE.

fractions = pd.read_csv("fractions_per_calmonth_and_degC.csv", index_col=0)


@functools.lru_cache()
def offtakevolume(delta_t):
    fractions_per_calmonth = fractions[f"{delta_t} degC"]
    i = offtakevolume_100.index
    frac = pd.Series(i.map(lambda ts: fractions_per_calmonth[ts.month]), i)
    offtakevolume_100 * frac

    cutoff = pd.Timestamp.now().floor("D").tz_localize("Europe/Berlin")
    s1 = (offtakevolume_100 * frac).w[i < cutoff]
    s2 = offtakevolume_100.w[i >= cutoff]
    return lb.PfLine({"w": pd.concat([s1, s2])})


#%% OFFTAKE TO TOTAL COST.


@functools.lru_cache()
def qhpfc(delta_p):
    cutoff = pd.Timestamp.now().floor("D").tz_localize("Europe/Berlin")
    s1 = unsourcedprice
    s2 = unsourcedprice + delta_p
    return lb.PfLine(
        {"p": pd.concat([s1.p[s1.index < cutoff], s2.p[s2.index >= cutoff]])}
    )


# \\lbfiler3\daten\Beschaffung\Planung\3Jahresplanung\2020\Ludwig\Ludwig Nachtspeicherheizung\20201007 5 JAHRESPLANUNG Ludwig NSP Versand.xlsx
salesprice = lb.PfLine(
    pd.Series(
        [0.9 * 43.43 + 0.1 * 42.51, 40.61],
        pd.date_range("2021-11", freq="MS", periods=2, tz="Europe/Berlin"),
    ).rename("p")
)


def pnl(delta_t, delta_p):
    pfs = lb.PfState(offtakevolume(delta_t), qhpfc(delta_p), sourced).changefreq("MS")
    offtake = -1 * pfs.offtake.volume
    cost = pfs.pnl_cost
    income = offtake * salesprice
    return pd.DataFrame(
        {
            ("offtake", "abs"): offtake.q,
            ("cost", "abs"): cost.r,
            ("cost", "rel"): cost.p,
            ("income", "abs"): income.r,
            ("income", "rel"): income.p,
            ("profit", "abs"): (income - cost).r,
            ("profit", "rel"): (income - cost).r / offtake.q,
        }
    )


# %% CREATE DATAFRAME

pnls = {}
for delta_t in tqdm(range(-3, 4)):
    for delta_p in range(-40, 50, 10):
        pnls[(delta_t, delta_p)] = pnl(delta_t, delta_p)[[("profit", "abs")]]

dez_result = (
    pd.Series({key: df[("profit", "abs")].iloc[1] for key, df in pnls.items()})
    .astype("pint[MEur]")
    .unstack()
)
nov_result = (
    pd.Series({key: df[("profit", "abs")].iloc[0] for key, df in pnls.items()})
    .astype("pint[MEur]")
    .unstack()
    .pint.dequantify()
)
dez_result.to_clipboard()
nov_result.to_clipboard()
# %
# %%
