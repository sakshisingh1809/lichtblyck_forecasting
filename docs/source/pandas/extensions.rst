Lichtblyck and pandas
#####################

The lichtblyck library uses pandas Dataframes and Series, especially as input or output for timeseries.

Standardization
***************


To this end, there are several assumptions and requirements concerning timeseries data, especially for the index:

* The index is a `pandas.DatetimeIndex`;
* it is gapless, with the `.freq` attribute set;
* the timezone is set;
* its values are left-bound.

A method :func:`lichtblyck.set_ts_index` is provided to coerce input `pandas.DataFrame` or `pandas.Series` objects into the specifications.

.. autofunction:: lichtblyck.set_ts_index

Extended methods
****************

When the `DatetimeIndex` conforms to these specifications, useful information can be calculated. To make this easier, the `pandas.DatetimeIndex` class is extended with several methods:


* `.ts_right`

  .. autofunction:: lichtblyck.tools.stamps.ts_right

* `.duration`

  .. autofunction:: lichtblyck.tools.stamps.duration
