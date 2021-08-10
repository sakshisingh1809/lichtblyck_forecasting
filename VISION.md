# lichtblyck

Repository with functions to do time-series analysis.

Developer: Ruud Wijtvliet (ruud.wijtvliet@lichtblick.de)

---

This document describes the vision for the `PfFrame` and `Portfolio` classes.

---

In short: 

* The `PfFrame` class is an extension of the `pandas` `DataFrame` class. Each instance stores related information about a single timeseries, most notably its volume information (in units of MW and/or MWh), and, if available, pricing information (in units of Eur and/or Eur/MWh).

* The `Portfolio` class is a collection. It stores other (sub-)portfolios and/or `PfFrame`s, and is able to aggregate the information correctly to higher levels

# Vision

The `Portfolio` object contains all information about a portfolio:

* Types of information:
  * Contracted and expected:
    * It contains timeseries with contracted volumes and prices, for offtake and procurement contracts;
    * It contains expected volumes and prices;
    * It is able to aggregate this information over all sub-portfolios, to a single timeseries of volumes and prices;
    * It is able to aggregate this information in the time domain, to a timeseries of month-, year-, or quarter-values;
    * It is able to show output this data in a human-readable way; as a table and as plots.
  * Uncertainty:
    * It is able to hold data about the uncertainty in the volume and price timeseries;
    * It is able to use these individual uncertainties to simulate the portfolio; to get various possible futures;

* Structure:
  * Classification:
    * Two types of portfolios:
      * leaf node
        * **does not** contain child `Portfolio`s
        * **does** contain an "own" `PfFrame`
      * internal node
        * **does** contain child `Portfolios`
        * **does not** contain "own" `PfFrame`
    * No 'mixed' portfolios
  * Sample structure of portfolios and their attributes:
    * LUD [`Portfolio`]
      * procured [`Portfolio`]
        * FWD [`PfFrame`]
        * DA [`PfFrame`]
        * ID [`PfFrame`]
      * offtake [`PfFrame`]
      * LUD_SIM [`Portfolio`]
        * procured ... etc

# Roadmap

## Step 1: Single Portfolio as normal DataFrame

Construct a DataFrame that contains non-redundant information. 

Columns: 

* `qs` (sourced MWh)
* `rs` (sourced Eur)
* `qo` (offtake MWh)
* `pu` (unhedged price Eur/MWh)

Index: DateTimeIndex.

## Step 2: Single Portfolio as custom object

Create a class `SinglePf` that has additional attributes to more easily work with portfolio data:

* Constructor with parameters `qs`, `rs`, `qo`, `pu`. Checks if data is consistent, i.e., if it uses same frequency etc. Stay close to `DataFrame` constructor if possible. Not all parameters must be supplied: e.g., only `qs` and `rs` may be supplied if this is a portfolio containing sourced volume but no offtake. TBD: save as individual series, or as dataframe?
* Read-only boolean properties `._hassourced`, `._hasofftake`, `._hasmarketprices`; to classify the portfolio. Calculated on-demand.
* Properties `.qs`, `.rs`, `.qo`, `.pu`, `.duration`: as originally specified. Return `Series`
* Properties `.qu`, `.ws`, `.wo`, `.wu`, `.ru`, `.ro`, `.ps`, `.po`: calculated on-demand, from the saved dataframe. Return `Series`.
* Unit tests

## Step 3: Add relevant methods

* Attribute `.changefreq()` to up- or downsample to a certain frequency. Returns `SingePf` object.
* Attribute `.__add__()`: check if `pu` is the same; Return `SingePf` object with summed `qs`, `rs`, `qo` Series. Assume 0 for non-overlapping timestamps? (--> Check how `df1 + df2` works, use same logic if it makes sense here.)
* Implement *setting* the series (`=`). When setting any of the series, always translate into setting `.qs`, `.rs`, `.qo`, or `.pu`, and save *those* values. Not all series may be set, e.g., `qu` may not be set once both `qs` and `qo` are set, as it's unclear, which of these 2 should be adjusted to remain consistent.
* To discuss: can we implement a `*=` and `+=` for the series?
* Unit tests
