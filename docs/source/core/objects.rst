#######
Objects
#######

The objects at the center of the library are `PfLine` and `PfState`.

* `PfLine` stores a timeseries containing volume information, price information, or both.
  For example: 

  * The forward price curve, in daily resolution, for tomorrow until the end of the frontyear+4. This `PfLine` only contains price information in terms of Eur/MWh.
  * The expected offtake volume of a certain portfolio, for the coming calender year, in quarterhourly resolution. This is a `PfLine` that only contains volume information. The volume in each timestamp can be retrieved by the user in units of energy (e.g., MWh) or in units of power (e.g., MW). 
  * The sourced hegde volume of the same portfolio, again for the coming calender year but in monthly resolution. This is a `PfLine` that, for each timestamp, contains a volume (the contracted volume, both as energy and as power), a price (for which the volume was contracted) and a revenue (in Eur, being the multiplication of the energy and the price).

  Not all information that can be retrieved by the user is stored; redundant information is discarded and recalculated whenever necessary. For the volume, for example, only the energy is stored. The power can be calculated by dividing the energy (in MWh) by the duration of the timestamp (in h).

  More information can be found here: :doc:`/core/pfline`.

* `PfState` stores several `PfLine` objects that all relate to the same portfolio. It stores information about the offtake in the portfolio; how much volume has currently been sourced and at what price; how much volume is consequently still unsourced and what is its current market price; what is the current best-guess for the procurement price of the portfolio (i.e., combining sourced and unsourced volumes to satisfy offtake), etc. 
 
  Here too, not all this information is stored; e.g. the unsourced volume timeseries can be calculated from the offtake and sourced volume timeseries, etc. 
 
  More information can be found here: :doc:`/core/pfstate`.