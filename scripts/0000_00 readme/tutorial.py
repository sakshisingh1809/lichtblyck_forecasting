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

# %% COMBINING OFFTAKE, SOURCED, AND FORWARD PRICES.

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
#            2023-01-01 00:00:00 +0100       -111.6         -28
#            2023-01-01 00:15:00 +0100       -104.2         -26
#            ..                                  ..          ..          ..            ..
#            2023-12-31 23:30:00 +0100        -95.1         -24
#            2023-12-31 23:45:00 +0100        -87.5         -22
# ─●────── pnl_cost
#  │         2023-01-01 00:00:00 +0100        111.6          28      106.07         2 960
#  │         2023-01-01 00:15:00 +0100        104.2          26       93.93         2 448
#  │         ..                                  ..          ..          ..            ..
#  │         2023-12-31 23:30:00 +0100         95.1          24       84.24         2 003
#  │         2023-12-31 23:45:00 +0100         87.5          22       78.89         1 726
#  ├────── sourced
#  │         2023-01-01 00:00:00 +0100         80.5          20       87.45         1 759
#  │         2023-01-01 00:15:00 +0100         80.5          20       87.45         1 759
#  │         ..                                  ..          ..          ..            ..
#  │         2023-12-31 23:30:00 +0100         73.0          18       76.94         1 404
#  │         2023-12-31 23:45:00 +0100         73.0          18       76.94         1 404
#  └────── unsourced
#            2023-01-01 00:00:00 +0100         31.2           8      154.17         1 201
#            2023-01-01 00:15:00 +0100         23.8           6      115.88           689
#            ..                                  ..          ..          ..            ..
#            2023-12-31 23:30:00 +0100         22.1           6      108.31           599
#            2023-12-31 23:45:00 +0100         14.5           4       88.68           322

# Let's see the forward price curve used:
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

# Or the hedge fraction, on a quarter level:
pfs.changefreq("QS").hedgefraction
# ts_left
# 2023-01-01 00:00:00+01:00    0.5973314694274244
# 2023-04-01 00:00:00+02:00    0.7105400540378203
# 2023-07-01 00:00:00+02:00    0.7762408103086229
# 2023-10-01 00:00:00+02:00    0.6497044002532851
# Freq: QS-JAN, Name: fraction, dtype: pint[]

# %% RISK CALCULATIONS

# Now, let's do some rudimentary risk calculations.

# We start with the `PfState` object from above.
# Here is the expected procurement price on a monthly level:
pfs.changefreq("MS").pnl_cost
# PfLine object with price and volume information.
# . Timestamps: first: 2023-01-01 00:00:00+01:00     timezone: Europe/Berlin
#                last: 2023-12-01 00:00:00+01:00         freq: <MonthBegin> (12 datapoints)
# . Children: 'sourced' (price and volume), 'unsourced' (price and volume)
#
#                                      w           q           p             r
#                                     MW         MWh     Eur/MWh           Eur
#
# 2023-01-01 00:00:00 +0100        139.6     103 833      138.50    14 381 219
# 2023-02-01 00:00:00 +0100        134.3      90 275      139.28    12 573 881
# 2023-03-01 00:00:00 +0100        124.6      92 551      125.05    11 573 857
# 2023-04-01 00:00:00 +0200        113.7      81 861       95.17     7 790 727
# 2023-05-01 00:00:00 +0200        102.7      76 380       92.51     7 065 694
# 2023-06-01 00:00:00 +0200         95.1      68 447       92.78     6 350 629
# 2023-07-01 00:00:00 +0200         90.7      67 490       89.59     6 046 138
# 2023-08-01 00:00:00 +0200         91.4      68 024       89.35     6 077 868
# 2023-09-01 00:00:00 +0200         96.1      69 201       94.31     6 526 163
# 2023-10-01 00:00:00 +0200        104.2      77 648      102.55     7 962 727
# 2023-11-01 00:00:00 +0100        111.6      80 373      113.23     9 100 758
# 2023-12-01 00:00:00 +0100        121.5      90 406      110.81    10 017 900

# So, based on current procurement and current forward curve prices, the January volume
# of 103 833 MWh is expected to cost 14 381 219 Eur, or 138.50 Eur/MWh

# Now, what will happen if the market prices go up by, let's say 50%? In that case, the price
# of the unsourced volume is affected, while the contracted volume stays equally expensive.
# The new price of the unsourced volume, i.e., the new forward curve, will be
new_prices = pfs.unsourcedprice * 1.5

# And the portfolio state, with these new prices, would be
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
# 2023-01-01 00:00:00 +0100        139.6     103 833      183.43    19 045 651
# 2023-02-01 00:00:00 +0100        134.3      90 275      183.65    16 579 118
# 2023-03-01 00:00:00 +0100        124.6      92 551      160.44    14 848 959
# 2023-04-01 00:00:00 +0200        113.7      81 861      117.20     9 594 357
# 2023-05-01 00:00:00 +0200        102.7      76 380      110.46     8 436 835
# 2023-06-01 00:00:00 +0200         95.1      68 447      108.65     7 436 830
# 2023-07-01 00:00:00 +0200         90.7      67 490      103.35     6 975 157
# 2023-08-01 00:00:00 +0200         91.4      68 024      103.16     7 017 129
# 2023-09-01 00:00:00 +0200         96.1      69 201      112.10     7 757 670
# 2023-10-01 00:00:00 +0200        104.2      77 648      126.55     9 826 437
# 2023-11-01 00:00:00 +0100        111.6      80 373      144.28    11 596 589
# 2023-12-01 00:00:00 +0100        121.5      90 406      142.75    12 905 616

# So, while the January volume has remained the same, its price has risen to 183.43 Eur/MWh.

# We can also compare the before and after portfolio prices
pfprice_before = pfs.changefreq("MS").pnl_cost.price
pfprice_after = new_pfs.changefreq("MS").pnl_cost.price
pricerise = pfprice_after / pfprice_before - 1
# ts_left
# 2023-01-01 00:00:00+01:00     0.3243
# 2023-02-01 00:00:00+01:00     0.3185
# 2023-03-01 00:00:00+01:00     0.2829
# 2023-04-01 00:00:00+02:00    0.2315
# 2023-05-01 00:00:00+02:00    0.1940
# 2023-06-01 00:00:00+02:00    0.1710
# 2023-07-01 00:00:00+02:00     0.1536
# 2023-08-01 00:00:00+02:00    0.1545
# 2023-09-01 00:00:00+02:00    0.1887
# 2023-10-01 00:00:00+02:00    0.2340
# 2023-11-01 00:00:00+01:00    0.2742
# 2023-12-01 00:00:00+01:00    0.2882
# Freq: MS, Name: fraction, dtype: pint[]

# As expected, the price rise is especially large in the months where the hedge fracion is low.


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
