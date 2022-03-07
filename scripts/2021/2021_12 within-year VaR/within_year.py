"""Calculate the value-at-risk for the tariff-fixed portfolios, based on current volatility."""

#%% IMPORTS AND SETUP.

from dataclasses import dataclass
import lichtblyck as lb


@dataclass
class WhatIf:
    quantile: float
    factor: float
    pfl: lb.PfLine = None


portfolio = {}

lb.belvis.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")

vola = 1  # fraction per calender year

#%% CALCULATIONS.

for name in ("B2C_HH", "B2C_P2H"):
    portfolio[name] = {}
    pfs = lb.portfolios.pfstate("power", name, "2022")

    # CURRENT SITUATION.
    reference = pfs.asfreq("MS")
    portfolio[name]["ref"] = WhatIf(0.5, 1, reference.pnl_cost)
    portfolio[name]["hedgefraction"] = reference.netposition.volume / -reference.offtake

    # WHAT IF.
    for label, quantile in {"falling": 0.05, "rising": 0.95}.items():
        factor = lb.analyse.multiplication_factor(vola, 7 / 365, quantile)
        scenario = pfs.set_unsourcedprice(pfs.unsourcedprice * factor).asfreq("MS")
        portfolio[name][label] = WhatIf(quantile, factor, scenario.pnl_cost)


# %%

name = "B2C_HH"
portfolio[name]["ref"].pfl.to_clipboard()
portfolio[name]["falling"].pfl.to_clipboard()
portfolio[name]["rising"].pfl.to_clipboard()
