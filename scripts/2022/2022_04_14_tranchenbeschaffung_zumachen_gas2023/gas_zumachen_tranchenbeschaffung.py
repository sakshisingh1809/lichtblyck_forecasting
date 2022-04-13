#%%

import lichtblyck as lb
import numpy as np
import pandas as pd

lb.belvis.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")

values = {}

pfname = "B2C_LEGACY"
start, end = ("2023", "2024")
ref = lb.portfolios.pfstate("gas", pfname, start, end)

#%%

for delta in (-0.5, -0.25, 0, 0.25, 0.5, 1):
    changed = (
        ref.set_unsourcedprice(ref.unsourcedprice * (1 + delta)).asfreq("AS").pnl_cost
    )
    break
