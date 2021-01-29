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

## Step 1

`PfFrame` 

* is able to correctly calculate `w`, `q`, `r`, and `p`, even if only two of (`w` or `q`), `r`, `p` are present in the dataframe.

`Portfolio` 

* is able to store an "own" `PfFrame` as well as children portfolios. 
* is able to create a `PfFrame` containing all levels below it.







*