# This tutorial will show some functionalities of the lichtblyck library and how to use it.

# %% IMPORTS.

# We start with importing the package, which is commonly given the `lb` alias.

import lichtblyck as lb

# And, because we want to fetch data from Belvis, let's authenticate there.
lb.belvis.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")


# %% DATA FROM BELVIS.

# If we want to get data exactly as it is found in Belvis, we can use the `lb.belvis` module.
# For example, to see how much offtake we expect for the power B2C portfolio in 2023, we can do'

offtake = lb.belvis.offtakevolume("power", "PKG", "2023")

# %% PYTHON BASICS.

# Before we continue using this object,, let's discuss some helpful standard python functionality that is not exclusive to the `lichtblyck` library.

# a) We can see what functions are available in a module using the `dir()` fuction.
# We can ignore the ones that start with an underscore.
dir(lb.belvis)
# ['__builtins__', '__cached__', '__doc__', '__file__',  '__loader__', '__name__', '__package__', '__path__',
# '__spec__', 'auth_with_password', 'auth_with_passwordfile', 'auth_with_token', 'connector', 'data', 'forward',
# 'offtakevolume', 'sourced', 'spot', 'unsourcedprice', 'update_cache_files']

# b) We can get help on modules and functions with the `help()` function.
# It shows us what the function does, and which the parameters it takes.
help(lb.belvis.offtakevolume)
# offtakevolume(commodity: str, pfid: str, ts_left: Union[str, datetime.datetime, pandas._libs.tslibs.timestamps.Timestamp] = None, ts_right: Union[str, datetime.datetime, pandas._libs.tslibs.timestamps.Timestamp] = None) -> lichtblyck.core.pfline.single.SinglePfLine
#     Get offtake (volume) for a certain portfolio from Belvis.
#
#     Parameters
#     ----------
#     commodity : {'power', 'gas'}
#     pfid : str
#         Belvis portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM').
#     ts_left : Union[str, dt.datetime, pd.Timestamp], optional
#         Start of delivery period.
#     ts_right : Union[str, dt.datetime, pd.Timestamp], optional
#         End of delivery period.
#
#     Returns
#     -------
#     PfLine
lb.belvis.offtakevolume()

# If we don't supply a value for a required parameter, or if we supply an incorrect value, we'll be notified:
incorrectfunctioncall = lb.belvis.offtakevolume("power", "B3C")
# ValueError: Parameter ``pfid`` must be one of PKG, WP, NSp, LUD, LUD_NSp, LUD_NSp_SiM, LUD_Stg, LUD_Stg_SiM, LUD_WP, LUD_WP_SiM, PK_SiM, PK_Neu_FLX, PK_Neu_FLX_SiM, PK_Neu_NSP, PK_Neu_NSP_SiM, PK_Neu_WP, PK_Neu_WP_SiM, GK, SBSG; got 'B3C'.

# %% BACK TO THE EXAMPLE.

# We have the offtake stored in the `offtake` object from above. We can inspect it and
# see that it contains volume informatio, in the resolution in which is present in Belvis
# (in this case, 15 minutes):
offtake
# PfLine object with volume information.
# . Timestamps: first: 2023-01-01 00:00:00+01:00     timezone: Europe/Berlin
#                last: 2023-12-31 23:45:00+01:00         freq: <15 * Minutes> (35040 datapoints)
# . Children: none
#
#                                      w           q
#                                     MW         MWh

# 2023-01-01 00:00:00 +0100        -89.9         -22
# 2023-01-01 00:15:00 +0100        -84.0         -21
# 2023-01-01 00:30:00 +0100        -78.4         -20
# ..                                  ..          ..
# 2023-12-31 23:15:00 +0100        -82.5         -21
# 2023-12-31 23:30:00 +0100        -76.6         -19
# 2023-12-31 23:45:00 +0100        -70.6         -18

# The object has many methods to manipulate it. For example, to change its frequency from
# 15 minutes to months or quarters, we can use the `.changefreq` method:
offtake.changefreq("MS")
# PfLine object with volume information.
# . Timestamps: first: 2023-01-01 00:00:00+01:00     timezone: Europe/Berlin
#                last: 2023-12-01 00:00:00+01:00         freq: <MonthBegin> (12 datapoints)
# . Children: none
#
#                                      w           q
#                                     MW         MWh
#
# 2023-01-01 00:00:00 +0100       -111.9     -83 234
# 2023-02-01 00:00:00 +0100       -107.7     -72 350
# 2023-03-01 00:00:00 +0100        -99.7     -74 064
# 2023-04-01 00:00:00 +0200        -90.8     -65 379
# 2023-05-01 00:00:00 +0200        -82.0     -60 987
# 2023-06-01 00:00:00 +0200        -75.9     -54 659
# 2023-07-01 00:00:00 +0200        -72.4     -53 883
# 2023-08-01 00:00:00 +0200        -73.1     -54 384
# 2023-09-01 00:00:00 +0200        -76.8     -55 261
# 2023-10-01 00:00:00 +0200        -83.3     -62 028
# 2023-11-01 00:00:00 +0100        -89.2     -64 193
# 2023-12-01 00:00:00 +0100        -97.0     -72 166

# We can also store the quarterly information to a variable, so we can reuse it:
quarters = offtake.changefreq("MS")

# Let's see what other methods we have.
# (The ones starting with an underscore I left out)
dir(offtake)
# ['available', 'changefreq', 'children', 'df', 'flatten', 'hedge_with', 'index', 'items',
# 'keys', 'kind', 'loc', 'p', 'plot', 'plot_to_ax', 'po', 'price', 'print', 'q', 'r', 'set_p',
# 'set_price', 'set_q', 'set_r', 'set_volume', 'set_w', 'summable', 'to_clipboard', 'to_excel',
# 'values', 'volume', 'w']

# `plot` sounds interesting. Let's see what it does.
help(offtake.plot)
# plot(cols: 'str' = 'wp') -> 'plt.Figure' method of lichtblyck.core.pfline.single.SinglePfLine instance
#     Plot one or more timeseries of the PfLine.
#
#     Parameters
#     ----------
#     cols : str, optional
#         The columns to plot. Default: plot volume `w` [MW] and price `p` [Eur/MWh]
#         (if available).
#
#     Returns
#     -------
#     plt.Figure
#         The figure object to which the series was plotted.

# Apparently, the only parameter (`cols`) is optional, so we don't have to provide a value.
offtake.plot()
# (plots a figure of the offtake in the original 15 min resolution)

# We can also chain the methods. So, we can first turn the data into monthly values, and then plot these:
offtake.changefreq("MS").plot()
# (plots a figure of the offtake in the monthly resolution)

# And if we want to do more analyses in Excel, we can copy-paste the data into excel with:
offtake.to_clipboard()
# Or alternatively, we can write it to an Excel file directly:
offtake.to_excel("offtake_of_power_PKG_in_2023.xlsx")

# %% MORE DATA FROM BELVIS

# Offtake is not all that matters - we also want to know what we have already sourced.
# This information is also stored in Belvis:
sourced = lb.belvis.sourced("power", "PKG", "2023")

# This object contains volume AND price information:
sourced
# PfLine object with price and volume information.
# . Timestamps: first: 2023-01-01 00:00:00+01:00     timezone: Europe/Berlin
#                last: 2023-12-31 23:45:00+01:00         freq: <15 * Minutes> (35040 datapoints)
# . Children: 'forward' (price and volume), 'spot' (price and volume)
#
#                                      w           q           p             r
#                                     MW         MWh     Eur/MWh           Eur
#
# 2023-01-01 00:00:00 +0100         57.8          14       79.50         1 149
# 2023-01-01 00:15:00 +0100         57.8          14       79.50         1 149
# 2023-01-01 00:30:00 +0100         57.8          14       79.50         1 149
# ..                                  ..          ..          ..            ..
# 2023-12-31 23:15:00 +0100         57.8          14       79.50         1 149
# 2023-12-31 23:30:00 +0100         57.8          14       79.50         1 149
# 2023-12-31 23:45:00 +0100         57.8          14       79.50         1 149

# All the same methods are available as for the offtake object, such as aggregating to monthly values and plotting.

# We can see above that this object has 2 'children': 'forward' and 'spot', which we can
# see individually as well. Because 2023 hasn't started yet, though, the spot volume is zero, with undetermined price:
sourced.spot
# PfLine object with price and volume information.
# . Timestamps: first: 2023-01-01 00:00:00+01:00     timezone: Europe/Berlin
#                last: 2023-12-31 23:45:00+01:00         freq: <15 * Minutes> (35040 datapoints)
# . Children: none
#
#                                      w           q           p             r
#                                     MW         MWh     Eur/MWh           Eur
#
# 2023-01-01 00:00:00 +0100          0.0           0                         0
# 2023-01-01 00:15:00 +0100          0.0           0                         0
# 2023-01-01 00:30:00 +0100          0.0           0                         0
# ..                                  ..          ..          ..            ..
# 2023-12-31 23:15:00 +0100          0.0           0                         0
# 2023-12-31 23:30:00 +0100          0.0           0                         0
# 2023-12-31 23:45:00 +0100          0.0           0                         0

# %% THE PFSTATE OBJECT

# Combining the offtake volume, the sourced volume and its price, and the current forward curve,
# we can get a complete picture of the portfolio. The object is called a portfolio state or `PfState`.

# Because Belvis is a bit of a mess, some data manipulation must be done. For example, there are 2 offtake curves
# for each portfolio: the '100%' offtake and the 'certain volume' offtake. These details are
# taken care of behind the scenes, and we can get the 'best picture' portfolio state from the
# `lb.portfolios.pfstate()` function:
pfs = lb.portfolios.pfstate("power", "PKG", "2023")

# This object has all the information we need, again in 15 minute resolution:
pfs
# PfState object.
# . Timestamps: first: 2023-01-01 00:00:00+01:00     timezone: Europe/Berlin
#                last: 2023-12-31 23:45:00+01:00         freq: <15 * Minutes> (35040 datapoints)
#                                                 w           q           p             r
#                                                MW         MWh     Eur/MWh           Eur
# ──────── offtake
#            2023-01-01 00:00:00 +0100        -90.8         -23
#            2023-01-01 00:15:00 +0100        -84.9         -21
#            ..                                  ..          ..          ..            ..
#            2023-12-31 23:30:00 +0100        -78.3         -20
#            2023-12-31 23:45:00 +0100        -72.2         -18
# ─●────── pnl_cost
#  │         2023-01-01 00:00:00 +0100         90.8          23      107.20         2 433
#  │         2023-01-01 00:15:00 +0100         84.9          21       91.49         1 941
#  │         ..                                  ..          ..          ..            ..
#  │         2023-12-31 23:30:00 +0100         78.3          20       87.68         1 715
#  │         2023-12-31 23:45:00 +0100         72.2          18       81.64         1 473
#  ├●───── sourced
#  ││        2023-01-01 00:00:00 +0100         57.8          14       79.50         1 149
#  ││        2023-01-01 00:15:00 +0100         57.8          14       79.50         1 149
#  ││        ..                                  ..          ..          ..            ..
#  ││        2023-12-31 23:30:00 +0100         57.8          14       79.50         1 149
#  ││        2023-12-31 23:45:00 +0100         57.8          14       79.50         1 149
#  │├───── forward
#  ││        2023-01-01 00:00:00 +0100         57.8          14       79.50         1 149
#  ││        2023-01-01 00:15:00 +0100         57.8          14       79.50         1 149
#  ││        ..                                  ..          ..          ..            ..
#  ││        2023-12-31 23:30:00 +0100         57.8          14       79.50         1 149
#  ││        2023-12-31 23:45:00 +0100         57.8          14       79.50         1 149
#  │└───── spot
#  │         2023-01-01 00:00:00 +0100          0.0           0                         0
#  │         2023-01-01 00:15:00 +0100          0.0           0                         0
#  │         ..                                  ..          ..          ..            ..
#  │         2023-12-31 23:30:00 +0100          0.0           0                         0
#  │         2023-12-31 23:45:00 +0100          0.0           0                         0
#  └────── unsourced
#            2023-01-01 00:00:00 +0100         33.0           8      155.74         1 284
#            2023-01-01 00:15:00 +0100         27.1           7      117.09           792
#            ..                                  ..          ..          ..            ..
#            2023-12-31 23:30:00 +0100         20.5           5      110.82           567
#            2023-12-31 23:45:00 +0100         14.4           4       90.27

# We can aggregate the values to other time resolutions using `.changefreq()`. For
# example, in quarters:
pfs.changefreq("QS")
# PfState object.
# . Timestamps: first: 2023-01-01 00:00:00+01:00     timezone: Europe/Berlin
#                last: 2023-10-01 00:00:00+02:00         freq: <QuarterBegin: startingMonth=1> (4 datapoints)
#                                                 w           q           p             r
#                                                MW         MWh     Eur/MWh           Eur
# ──────── offtake
#            2023-01-01 00:00:00 +0100       -107.6    -232 338
#            2023-04-01 00:00:00 +0200        -84.2    -183 819
#            2023-07-01 00:00:00 +0200        -75.5    -166 662
#            2023-10-01 00:00:00 +0200        -91.8    -202 838
# ─●────── pnl_cost
#  │         2023-01-01 00:00:00 +0100        107.6     232 338      132.49    30 781 290
#  │         2023-04-01 00:00:00 +0200         84.2     183 819       93.60    17 205 147
#  │         2023-07-01 00:00:00 +0200         75.5     166 662       90.80    15 132 764
#  │         2023-10-01 00:00:00 +0200         91.8     202 838      108.47    22 001 072
#  ├●───── sourced
#  ││        2023-01-01 00:00:00 +0100         61.5     132 824       80.16    10 646 882
#  ││        2023-04-01 00:00:00 +0200         61.5     134 269       80.15    10 761 755
#  ││        2023-07-01 00:00:00 +0200         61.4     135 656       80.14    10 872 033
#  ││        2023-10-01 00:00:00 +0200         61.4     135 714       80.14    10 876 628
#  │├───── forward
#  ││        2023-01-01 00:00:00 +0100         61.5     132 824       80.16    10 646 882
#  ││        2023-04-01 00:00:00 +0200         61.5     134 269       80.15    10 761 755
#  ││        2023-07-01 00:00:00 +0200         61.4     135 656       80.14    10 872 033
#  ││        2023-10-01 00:00:00 +0200         61.4     135 714       80.14    10 876 628
#  │└───── spot
#  │         2023-01-01 00:00:00 +0100          0.0           0                         0
#  │         2023-04-01 00:00:00 +0200          0.0           0                         0
#  │         2023-07-01 00:00:00 +0200          0.0           0                         0
#  │         2023-10-01 00:00:00 +0200          0.0           0                         0
#  └────── unsourced
#            2023-01-01 00:00:00 +0100         46.1      99 514      202.33    20 134 408
#            2023-04-01 00:00:00 +0200         22.7      49 550      130.04     6 443 392
#            2023-07-01 00:00:00 +0200         14.0      31 006      137.42     4 260 732
#            2023-10-01 00:00:00 +0200         30.4      67 124      165.73    11 124 444

# Because we are looking at a delivery period that is fully in the future, the spot volume
# is 0, and the sourced volume consists entirely of forward volume.

# There is still unsourced volume. We can see how much volume is sourced, as a fraction of
# the offtake volume, using the `.hedgefraction` property. On a 15-minute resolution, this
# makes little sense, so let's do in on month level:
pfs.changefreq("MS").hedgefraction
# ts_left
# 2023-01-01 00:00:00+01:00    0.5435665353270686
# 2023-02-01 00:00:00+01:00    0.5643945482052004
# 2023-03-01 00:00:00+01:00    0.6103273987176215
# 2023-04-01 00:00:00+02:00     0.664881283738455
# 2023-05-01 00:00:00+02:00    0.7402294438967199
# 2023-06-01 00:00:00+02:00    0.7977566032048398
# 2023-07-01 00:00:00+02:00    0.8313065233261596
# 2023-08-01 00:00:00+02:00    0.8271321090968647
# 2023-09-01 00:00:00+02:00    0.7841339384580529
# 2023-10-01 00:00:00+02:00    0.7226482354294849
# 2023-11-01 00:00:00+01:00    0.6755733891337624
# 2023-12-01 00:00:00+01:00    0.6173544145698892
# Freq: MS, Name: fraction, dtype: pint[]


# The unsourced volume is valued using the forward price curve. Let's see the prices that are used:
pfs.unsourcedprice
# PfLine object with price information.
# . Timestamps: first: 2023-01-01 00:00:00+01:00     timezone: Europe/Berlin
#                last: 2023-12-31 23:45:00+01:00         freq: <15 * Minutes> (35040 datapoints)
# . Children: none
#
#                                     p
#                               Eur/MWh
#
# 2023-01-01 00:00:00 +0100      154.17
# 2023-01-01 00:15:00 +0100      115.88
# 2023-01-01 00:30:00 +0100       96.16
# ..                                 ..
# 2023-12-31 23:15:00 +0100      118.66
# 2023-12-31 23:30:00 +0100      108.31
# 2023-12-31 23:45:00 +0100       88.68

# And let's turn these into 'peak' and 'offpeak' values with the `.po()` function. For
# example, the month prices:
pfs.unsourcedprice.po("MS")
#                               peak              offpeak
#                           duration         p   duration         p
# ts_left
# 2023-01-01 00:00:00+01:00    312.0    229.55      432.0    158.20
# 2023-02-01 00:00:00+01:00    288.0    238.54      384.0    161.91
# 2023-03-01 00:00:00+01:00    324.0    208.38      419.0    144.71
# 2023-04-01 00:00:00+02:00    300.0    132.99      420.0    104.27
# 2023-05-01 00:00:00+02:00    324.0    129.61      420.0     99.45
# 2023-06-01 00:00:00+02:00    312.0    142.09      408.0    105.46
# 2023-07-01 00:00:00+02:00    312.0    130.97      432.0    103.71
# 2023-08-01 00:00:00+02:00    324.0    127.74      420.0     99.81
# 2023-09-01 00:00:00+02:00    312.0    144.89      408.0    111.01
# 2023-10-01 00:00:00+02:00    312.0    171.35      433.0    116.55
# 2023-11-01 00:00:00+01:00    312.0    198.45      408.0    125.48
# 2023-12-01 00:00:00+01:00    312.0    187.20      432.0    116.03


# %% HEDGING

# Let's calculate how much we'd need to buy to fully hedge the portfolio on a year level.
to_buy = pfs.hedge_of_unsourced("val", freq="AS")  #'val' for value hedge
# Let's store the portfolio state, in case we actually did this, in a new variable:
pfs2 = pfs.add_sourced(to_buy)

# Let's verify that the hedge fraction of this portfolio is indeed close to 100%:
pfs2.changefreq("MS").hedgefraction
# ts_left
# 2023-01-01 00:00:00+01:00    0.8380803170717684
# 2023-02-01 00:00:00+01:00    0.8704356381443035
# 2023-03-01 00:00:00+01:00    0.9428999890953694
# 2023-04-01 00:00:00+02:00     1.022450401567433
# 2023-05-01 00:00:00+02:00    1.1435182441538607
# 2023-06-01 00:00:00+02:00    1.2317504743889172
# 2023-07-01 00:00:00+02:00    1.2792165659814116
# 2023-08-01 00:00:00+02:00    1.2777668665794024
# 2023-09-01 00:00:00+02:00    1.2082834450683304
# 2023-10-01 00:00:00+02:00    1.1141274222467106
# 2023-11-01 00:00:00+01:00    1.0430974049065562
# 2023-12-01 00:00:00+01:00    0.9499865236709002
# Freq: MS, Name: fraction, dtype: pint[]

# %% PORTFOLIO PRICES

# We use with the `PfState` objects `pfs` and `pfs2` from above.

# Here is the expected procurement price on a monthly level for the first portfolio:
pfs.changefreq("MS").pnl_cost
# PfLine object with price and volume information.
# . Timestamps: first: 2023-01-01 00:00:00+01:00     timezone: Europe/Berlin
#                last: 2023-12-01 00:00:00+01:00         freq: <MonthBegin> (12 datapoints)
# . Children: 'sourced' (price and volume), 'unsourced' (price and volume)
#
#                                      w           q           p             r
#                                     MW         MWh     Eur/MWh           Eur
#
# 2023-01-01 00:00:00 +0100        113.1      84 116      136.89    11 514 308
# 2023-02-01 00:00:00 +0100        108.9      73 200      137.59    10 071 445
# 2023-03-01 00:00:00 +0100        101.0      75 022      122.57     9 195 536
# 2023-04-01 00:00:00 +0200         92.1      66 310       95.64     6 341 851
# 2023-05-01 00:00:00 +0200         83.2      61 935       92.58     5 733 831
# 2023-06-01 00:00:00 +0200         77.2      55 575       92.30     5 129 465
# 2023-07-01 00:00:00 +0200         73.7      54 852       89.33     4 899 935
# 2023-08-01 00:00:00 +0200         74.5      55 428       89.15     4 941 483
# 2023-09-01 00:00:00 +0200         78.3      56 383       93.85     5 291 345
# 2023-10-01 00:00:00 +0200         85.0      63 351      101.94     6 457 948
# 2023-11-01 00:00:00 +0100         91.1      65 626      112.22     7 364 833
# 2023-12-01 00:00:00 +0100         99.3      73 862      110.72     8 178 291

# So, based on current procurement and current forward curve prices, the January volume
# of 103 833 MWh is expected to cost 14 381 219 Eur, or 138.50 Eur/MWh

# For the other portfolio, the monthly values are different due to the procurement of
# year products. Its values for the individual months are
pfs2.changefreq("MS").pnl_cost
# PfLine object with price and volume information.
# . Timestamps: first: 2023-01-01 00:00:00+01:00     timezone: Europe/Berlin
#                last: 2023-12-01 00:00:00+01:00         freq: <MonthBegin> (12 datapoints)
# . Children: 'sourced' (price and volume), 'unsourced' (price and volume)
#
#                                      w           q           p             r
#                                     MW         MWh     Eur/MWh           Eur
#
# 2023-01-01 00:00:00 +0100        113.1      84 116      122.70    10 320 647
# 2023-02-01 00:00:00 +0100        108.9      73 200      120.92     8 851 508
# 2023-03-01 00:00:00 +0100        101.0      75 022      112.15     8 414 043
# 2023-04-01 00:00:00 +0200         92.1      66 310      104.71     6 943 385
# 2023-05-01 00:00:00 +0200         83.2      61 935      104.82     6 491 748
# 2023-06-01 00:00:00 +0200         77.2      55 575      101.46     5 638 629
# 2023-07-01 00:00:00 +0200         73.7      54 852      101.30     5 556 431
# 2023-08-01 00:00:00 +0200         74.5      55 428      103.15     5 717 633
# 2023-09-01 00:00:00 +0200         78.3      56 383      100.89     5 688 496
# 2023-10-01 00:00:00 +0200         85.0      63 351      102.44     6 489 665
# 2023-11-01 00:00:00 +0100         91.1      65 626      106.15     6 966 198
# 2023-12-01 00:00:00 +0100         99.3      73 862      108.88     8 041 890

# In this portfolio, the January volume is of course the same with 103 833 MWh, but the
# expected price is 122.70 Eur/MWh.

# On a yearly level, they are identical - which they should - with a price of 108.34 Eur/MWh:
pfs.changefreq("AS").pnl_cost
pfs2.changefreq("AS").pnl_cost
# PfLine object with price and volume information.
# . Timestamps: first: 2023-01-01 00:00:00+01:00     timezone: Europe/Berlin
#                last: 2023-01-01 00:00:00+01:00         freq: <YearBegin: month=1> (1 datapoints)
# . Children: 'sourced' (price and volume), 'unsourced' (price and volume)
#
#                                      w           q           p             r
#                                     MW         MWh     Eur/MWh           Eur
#
# 2023-01-01 00:00:00 +0100         89.7     785 657      108.34    85 120 273

# %% WHAT-IF

# Now, let's do some rudimentary risk calculations.
# What will happen if the market prices go up by, let's say 50%? In that case, the price
# of the unsourced volume is affected, while the contracted volume stays equally expensive.
# The new price of the unsourced volume, i.e., the new forward curve, will be
new_prices = pfs.unsourcedprice * 1.5

# Starting with the first portfolio: the portfolio state, with these new prices, would be
new_pfs = pfs.set_unsourcedprice(new_prices)

# We can again see the monthly expected procurement price in that case:
new_pfs.changefreq("MS").pnl_cost
# PfLine object with price and volume information.
# . Timestamps: first: 2023-01-01 00:00:00+01:00     timezone: Europe/Berlin
#                last: 2023-12-01 00:00:00+01:00         freq: <MonthBegin> (12 datapoints)
# . Children: 'sourced' (price and volume), 'unsourced' (price and volume)
#
#                                      w           q           p             r
#                                     MW         MWh     Eur/MWh           Eur
#
# 2023-01-01 00:00:00 +0100        113.1      84 116      183.55    15 439 214
# 2023-02-01 00:00:00 +0100        108.9      73 200      183.76    13 451 513
# 2023-03-01 00:00:00 +0100        101.0      75 022      159.39    11 957 766
# 2023-04-01 00:00:00 +0200         92.1      66 310      116.83     7 746 844
# 2023-05-01 00:00:00 +0200         83.2      61 935      109.19     6 762 911
# 2023-06-01 00:00:00 +0200         77.2      55 575      106.47     5 917 088
# 2023-07-01 00:00:00 +0200         73.7      54 852      100.69     5 523 243
# 2023-08-01 00:00:00 +0200         74.5      55 428      100.57     5 574 389
# 2023-09-01 00:00:00 +0200         78.3      56 383      109.35     6 165 497
# 2023-10-01 00:00:00 +0200         85.0      63 351      123.95     7 852 377
# 2023-11-01 00:00:00 +0100         91.1      65 626      141.26     9 270 140
# 2023-12-01 00:00:00 +0100         99.3      73 862      141.36    10 440 777

# So, the January price has risen by almost 50 Eur/MWh.

# We can also calculate the price change explicitly:
new_pfs.changefreq("MS").pnl_cost.p - pfs.changefreq("MS").pnl_cost.p
# ts_left
# 2023-01-01 00:00:00+01:00     46.660
# 2023-02-01 00:00:00+01:00     46.175
# 2023-03-01 00:00:00+01:00     36.818
# 2023-04-01 00:00:00+02:00     21.188
# 2023-05-01 00:00:00+02:00     16.615
# 2023-06-01 00:00:00+02:00     14.172
# 2023-07-01 00:00:00+02:00     11.363
# 2023-08-01 00:00:00+02:00     11.418
# 2023-09-01 00:00:00+02:00     15.503
# 2023-10-01 00:00:00+02:00     22.011
# 2023-11-01 00:00:00+01:00     29.032
# 2023-12-01 00:00:00+01:00     30.631
# Freq: MS, Name: p, dtype: pint[Eur/MWh]

# As expected, the price rise is especially large in the months where the hedge fraction
# is low (e.g. in the winter), and small where it is high (e.g. in the summer) .


# Repeating this for the other portfolio, which was fully hedged, we expect a much smaller price
# increase. Let's verify:
new_pfs2 = pfs2.set_unsourcedprice(new_prices)
new_pfs2.changefreq("MS").pnl_cost.p - pfs2.changefreq("MS").pnl_cost.p
# ts_left
# 2023-01-01 00:00:00+01:00      18.405
# 2023-02-01 00:00:00+01:00      15.835
# 2023-03-01 00:00:00+01:00       7.567
# 2023-04-01 00:00:00+02:00       0.244
# 2023-05-01 00:00:00+02:00      -6.415
# 2023-06-01 00:00:00+02:00     -12.565
# 2023-07-01 00:00:00+02:00     -14.634
# 2023-08-01 00:00:00+02:00     -14.153
# 2023-09-01 00:00:00+02:00     -11.392
# 2023-10-01 00:00:00+02:00      -5.860
# 2023-11-01 00:00:00+01:00      -0.526
# 2023-12-01 00:00:00+01:00       5.955
# Freq: MS, Name: p, dtype: pint[Eur/MWh]

# The January price has increased only by 20 Eur/MWh, and more importantly, some months
# have a price increase while others have a price decrease.


# %% TEMPERATURE-DEPENDENT LOAD PROFILES

# Another topic we often deal with is temperature-dependent offtake. Let's see how the
# library can help us there.

# Historic temperature profiles can be obtained from the `tmpr()` function in the `lb.tpmr.hist` module:
temperatures = lb.tmpr.hist.tmpr()

# This stores the daily temperatures of 15 German climate zones into a dataframe. Note that not
# not for every day and climate zone, a temperature is available, with missing values indicated with 'NaN':
temperatures
#                            t_1  t_2  t_3  t_4  t_5  t_6  t_7  t_8  t_9  t_10  t_11  t_12  t_13  t_14  t_15
# ts_left
# 1917-01-01 00:00:00+01:00  NaN  NaN  NaN  5.1  NaN  NaN  NaN  NaN  NaN   NaN   0.6   NaN   NaN   NaN   NaN
# 1917-01-02 00:00:00+01:00  NaN  NaN  NaN  6.1  NaN  NaN  NaN  NaN  NaN   NaN   0.4   NaN   NaN   NaN   NaN
# 1917-01-03 00:00:00+01:00  NaN  NaN  NaN  4.5  NaN  NaN  NaN  NaN  NaN   NaN   0.4   NaN   NaN   NaN   NaN
# 1917-01-04 00:00:00+01:00  NaN  NaN  NaN  6.4  NaN  NaN  NaN  NaN  NaN   NaN  -1.1   NaN   NaN   NaN   NaN
# ...
# 2020-12-28 00:00:00+01:00  3.6  3.1  3.4  1.9  3.5  0.5  1.1 -0.1  0.7  -1.3  -4.9   4.3   0.8   0.1  -0.2
# 2020-12-29 00:00:00+01:00  2.2  2.3  2.6  1.4  2.2 -0.2  0.3 -0.8  1.5   0.2  -4.0   3.7  -0.2   0.4  -3.8
# 2020-12-30 00:00:00+01:00  4.2  4.0  3.9  1.5  2.9  0.3  0.4 -0.3  1.8  -0.3  -3.9   3.8   0.2  -0.5  -3.7
# 2020-12-31 00:00:00+01:00  3.2  3.4  3.4  2.0  2.0 -0.9 -0.1 -1.2  1.2   0.1  -4.8   2.4  -0.4  -1.0  -6.5

# We can select the Hamburg climate zone ('t_3') and only keep days with a valid temperature:
temperatures["t_3"].dropna()
# ts_left
# 1936-01-01 00:00:00+01:00    7.3
# 1936-01-02 00:00:00+01:00    6.7
# 1936-01-03 00:00:00+01:00    4.4
# 1936-01-04 00:00:00+01:00    4.5
# 1936-01-05 00:00:00+01:00    2.6
#                             ...
# 2020-12-27 00:00:00+01:00    3.8
# 2020-12-28 00:00:00+01:00    3.4
# 2020-12-29 00:00:00+01:00    2.6
# 2020-12-30 00:00:00+01:00    3.9
# 2020-12-31 00:00:00+01:00    3.4
# Freq: D, Name: t_3, Length: 31047, dtype: float64

# But let's select data from 1990 or later, and store them in the variable `t`:
t = temperatures["t_3"].loc["1990":]

# In summer, it's warm; in winter, it's cold; and daily deviations from the trend are large,
# as can be seen when plotting the data:
t.plot()

# For our purposes, we'd like to translate the temperatures into an offtake, according to
# a certain temperature load profile. These TLPs are available in the `lb.tlp` module.
# Let's start with a P2H profile. For this, the `lb.tlp.power.fromsource()` function is
# provided. (remember we can always use the `help()` function in case usage are unclear)
tlp = lb.tlp.power.fromsource("avacon_hz0", spec=1000)

# The time- and temperature-dependence of this profile can be plotted with
lb.tlp.plot.vs_time(tlp)

# and be passing it the temperature profile we just created, we can see how much this
# customer has consumed:
offtake_in_MW = tlp(t)

# Although the temperature data has daily values, the offtake has a 15-minute resolution.
# The reason is that the P2H profiles provide a daily curve in 15-minute resolution for
# each temperature:
offtake_in_MW
# ts_left
# 1990-01-01 00:00:00+01:00    2.256239
# 1990-01-01 00:15:00+01:00    2.322071
# 1990-01-01 00:30:00+01:00    2.387903
# 1990-01-01 00:45:00+01:00    2.453735
# 1990-01-01 01:00:00+01:00    2.438902
#                                ...
# 2020-12-31 22:45:00+01:00    1.486237
# 2020-12-31 23:00:00+01:00    1.902884
# 2020-12-31 23:15:00+01:00    1.906835
# 2020-12-31 23:30:00+01:00    1.910786
# 2020-12-31 23:45:00+01:00    1.892096
# Freq: 15T, Name: w, Length: 1087008, dtype: float64

# Using this offtake timeseries, we can again create a similar object to the one we started with:
offtake = lb.PfLine(offtake_in_MW)

# To make the data a bit more handlable and reduce the number of datapoints (there are > 1 million
# quarterhours between 1990 and 2020), we can aggregate to daily, monthly, quarterly or yearly values:
yearly = offtake.changefreq("AS")

# Which we can then again plot, for example as bars. 1996 and 2010 have extremely high consumption
# due to their low temperatures.
yearly.q.plot(kind="bar")
