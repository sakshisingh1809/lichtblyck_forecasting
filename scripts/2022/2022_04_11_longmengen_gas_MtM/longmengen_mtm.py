"""Calculate the value (market and mtm) of long positions in gas portfolio."""
# %%


import lichtblyck as lb
import numpy as np
import pandas as pd

lb.belvis.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")


def openpositions_and_mtmchange(pfname, start, end, freq):
    pfs = lb.portfolios.pfstate("gas", pfname, start, end)
    sourced = pfs.asfreq(freq).sourced

    openpos_now = -pfs.hedge_of_unsourced(freq).asfreq(freq)
    mtm_now = (openpos_now.price - sourced.price) * openpos_now.volume

    pfs2 = pfs.set_unsourcedprice(pfs.unsourcedprice * 0.5)
    openpos_fut = -pfs2.hedge_of_unsourced(freq).asfreq(freq)
    mtm_fut = (openpos_fut.price - sourced.price) * openpos_fut.volume

    df = pd.DataFrame(
        {
            ("open", "q"): openpos_now.q,
            ("sourced", "price"): sourced.p,
            ("now", "marketprice"): openpos_now.p,
            ("now", "marketvalue"): openpos_now.r,
            ("now", "mtm"): mtm_now.r,
            ("fut", "marketprice"): openpos_fut.p,
            ("fut", "marketvalue"): openpos_fut.r,
            ("fut", "mtm"): mtm_fut.r,
        }
    )

    return df


dfs2022 = {}
for pfname in ("B2C_LEGACY", "B2C_NEW", "B2B_CONTI", "B2B_BTB", "B2B_RLM"):
    dfs2022[pfname] = openpositions_and_mtmchange(pfname, "2022-06", "2023", "MS")
pd.concat(dfs2022).pint.dequantify().tz_localize(None, 0, 1).reorder_levels(
    [1, 0]
).sort_index().to_excel("2022roy.xlsx")

dfs2023 = {}
for pfname in ("B2C_NEW", "B2B_CONTI", "B2B_BTB", "B2B_RLM"):
    dfs2023[pfname] = openpositions_and_mtmchange(pfname, "2023", "2026", "QS")
pd.concat(dfs2023).pint.dequantify().tz_localize(None, 0, 1).reorder_levels(
    [1, 0]
).sort_index().to_excel("2023-2026.xlsx")

# %%
