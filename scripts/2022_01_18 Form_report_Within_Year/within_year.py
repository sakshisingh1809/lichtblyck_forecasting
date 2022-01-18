"""Create overview of within-year positions of B2C portfolios."""

import lichtblyck as lb

lb.belvis.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")

lb.portfolios.pfstate("power", "B2C_HH")
