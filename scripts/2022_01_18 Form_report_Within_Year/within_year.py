# %%

import lichtblyck as lb
import pandas as pd

lb.belvis.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")
pfname = "B2C_P2H_NEW"
pfs = lb.portfolios.pfstate("power", pfname, "2022", "2026")

# %%

freq = "AS"

aggpfs = pfs.changefreq(freq)
aggprices = pfs.unsourcedprice.po(freq)

dfs = {
    "offtake": -1 * aggpfs.offtake.df("q"),
    "hedged": pd.DataFrame({"fraction": aggpfs.hedgefraction, "p": aggpfs.sourced.p}),
    "current_market_offpeak": aggprices["offpeak"][["p"]],
    "current_market_peak": aggprices["peak"][["p"]],
    "portfolioprice": aggpfs.pnl_cost.df("p"),
}

big_df = pd.concat(dfs, axis=1)
big_df.pint.dequantify().tz_localize(None).to_excel(f"{pfname}.xlsx")
# %% Check one month in more detail.

mask = pfs.offtake.index <= "2022-02"
pfs_jan = lb.PfState.from_series(
    pu=pfs.unsourcedprice.p.loc[mask],
    qo=pfs.offtake.q.loc[mask],
    qs=pfs.sourced.q.loc[mask],
    rs=pfs.sourced.r.loc[mask],
)

# %%
