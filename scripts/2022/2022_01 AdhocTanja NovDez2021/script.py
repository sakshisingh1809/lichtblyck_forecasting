import lichtblyck as lb
import pandas as pd

lb.belvis.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")

pfl = lb.portfolios.pfstate("power", "B2C_P2H", "2021-10", "2022")
pfl.changefreq("D").to_excel("data2.xlsx")

# %% December sensititivy for p2h.

p_old = pd.read_csv("QHPFC_FSP2020.csv", sep=";")
p_old = p_old[p_old.columns[-1]].str.replace(",", ".").astype(float)
p_old.index = pd.date_range(
    "2021-10", "2022", freq="15T", closed="left", tz="Europe/Berlin"
)
p_old = lb.PfLine({"p": p_old})

pfl = lb.portfolios.pfstate("power", "B2C_P2H", "2021-10", "2022")
pfl = pfl.set_unsourcedprice(p_old)
sensitivity_fraction_per_degC = -0.068965576707521
dfs = {}
for change_tmpr_degC in (0, -1, -2, -3):
    change_offtake_fraction = change_tmpr_degC * sensitivity_fraction_per_degC
    offtake_factor = 1 + change_offtake_fraction
    pfl2 = pfl.set_offtakevolume(pfl.offtake.volume * offtake_factor)
    for change_price_fraction in (0, 0.25, 0.5, 1, 2, 4):
        price_factor = 1 + change_price_fraction
        pfl3 = pfl2.set_unsourcedprice(pfl.unsourcedprice * price_factor)
        df = pfl3.changefreq("QS").df()
        df.index = df.index.tz_localize(None)
        dfs[(change_tmpr_degC, change_price_fraction)] = df
result = pd.concat(dfs).pint.dequantify()
result.to_excel("sensitivity_results_starting_from_old_pfc_inQS.xlsx")

# %%
