#%%

import lichtblyck as lb

lb.belvis.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")
pf = lb.portfolios.pfstate("power", "", "2022")

#%%

d = pf.asfreq("D")

#%%

p = lb.PfLine.from_belvis_forwardpricecurve("power", "2021")
# %%
