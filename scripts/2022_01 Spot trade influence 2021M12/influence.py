#%%

import lichtblyck as lb

lb.belvis.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")
pf = lb.portfolios.pfstate("power", "B2C_P2H", "2021-12", "2022")

# %%

d = pf.changefreq("D")
d.to_clipboard()

# %%

p = lb.PfLine.from_belvis_forwardpricecurve("power", "2021")
# %%
