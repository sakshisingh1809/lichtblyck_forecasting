#%%

import lichtblyck as lb
import numpy as np
import pandas as pd

lb.belvis.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")

values = {}

for pfname in ("B2C_LEGACY", "B2B_BTB", "B2B_RLM", "B2C_NEW", "B2B_CONTI"):
    for (start, end) in (("2022-05", "2023"), ("2023", "2025")):
        ref = lb.portfolios.pfstate("gas", pfname, start, end)

        openpos_now = ref.netposition
        openpos_pricedrop = openpos_now.set_price(ref.unsourcedprice * 0.5)

        openpos_now_cumulvals = openpos_now.df().apply(np.sum)
        openpos_pricedrop_cumulvals = openpos_pricedrop.df().apply(np.sum)

        openpos_pricedrop_cumulvals - openpos_now_cumulvals

        values[(pfname, start)] = {
            "now": {"q": openpos_now_cumulvals.q.m, "r": openpos_now_cumulvals.r.m},
            "pricedrop": {
                "r": openpos_pricedrop_cumulvals.r.m,
                "delta": openpos_pricedrop_cumulvals.r.m - openpos_now_cumulvals.r.m,
            },
        }

# %%

df = pd.DataFrame({k1: pd.DataFrame(v1).stack() for k1, v1 in values.items()})
df = df.T
# %%
