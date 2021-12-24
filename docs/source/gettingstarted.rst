Getting started
###############

This library is used for monitoring portfolios (getting the current best-picture) and doing risk analyses (possible future developments).

As a data source, Belvis is commonly used. We first need to authorize with its REST-API:
   
   >>> import lichtblyck as lb
   >>> lb.belvis.auth_with_password('Ruud.Wijtvliet', 'my_long_and_5tr0ng_password')

Then, we can get portfolio state of several pre-defined portfolios. The available ones are found here:

.. code-block::

   >>> lb.portfolios.PFNAMES
   {'power': ['PKG', 'NSP', 'WP', 'LUD_STG', 'LUD_NSP', 'LUD_WP', 'B2C_P2H', 'B2C_HH'], 'gas': []}

And 

TODO