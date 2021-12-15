#%%

import lichtblyck as lb

lb.belvis.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")

pfs = lb.portfolios.pfstate("power", "B2C_P2H", "2022", "2022-04")


#%%
pfs.changefreq("D").to_excel("b2c_p2h_2022q1.xlsx")


# %%
