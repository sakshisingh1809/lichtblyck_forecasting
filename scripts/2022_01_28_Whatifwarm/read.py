import pandas as pd

filenames = {
    "gas": "gas_offtakescalingfactors_per_calmonth_and_degC.csv",
    "p2h": "p2h_offtakescalingfactors_per_calmonth_and_degC.csv",
}

ss = {}
for pf, path in filenames.items():
    table = pd.read_csv(path, index_col=0)
    sensitivity = table["1 degC"] - 1
    ss[pf] = sensitivity

sensitivity = pd.DataFrame(ss)

# %% INFLUENCE ON PF RESULT


import lichtblyck as lb

lb.belvis.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")

pfs = lb.portfolios.pfstate("power", "B2C_P2H_LEGACY", "2022", "2023")
# %%


current_costs = pfs.changefreq("MS").pnl_cost
current_offtake = pfs.offtake.volume


factor_when_warmer = sensitivity["p2h"] + 1
factorseries = pd.Series(
    current_offtake.index.map(lambda ts: factor_when_warmer[ts.month]),
    current_offtake.index,
)
warmer_offtake = current_offtake * factorseries
warmer_costs = pfs.set_offtakevolume(warmer_offtake).changefreq("MS").pnl_cost

warmer_income = (pfs.pnl_cost.price * -warmer_offtake).changefreq("MS")

# Change in PNL:
change_in_profit = warmer_income - warmer_costs

# %%
