import lichtblyck as lb
import pandas as pd

lb.belvis.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")

pfl = lb.portfolios.pfstate("power", "B2C_P2H", "2021-10", "2022")
pfl.changefreq("D").to_excel("data.xlsx")

# %% December sensititivy for p2h.

pfl = lb.portfolios.pfstate("power", "B2C_P2H", "2021-12", "2022")
sensitivity_fraction_per_degC = -0.068965576707521
dfs = {}
for change_tmpr_degC in (0, -1, -2):
    change_offtake_fraction = change_tmpr_degC * sensitivity_fraction_per_degC
    offtake_factor = 1 + change_offtake_fraction
    pfl2 = pfl.set_offtakevolume(pfl.offtake.volume * offtake_factor)
    for change_price_fraction in (0, 0.1, 0.20):
        price_factor = 1 + change_price_fraction
        pfl3 = pfl2.set_unsourcedprice(pfl.unsourcedprice * price_factor)
        df = pfl3.changefreq("MS").df()
        df.index = df.index.tz_localize(None)
        dfs[(change_tmpr_degC, change_price_fraction)] = df
result = pd.concat(dfs).pint.dequantify()
result.to_excel("sensitivity_results.xlsx")

# %%
