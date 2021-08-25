"""
Dataframe-like class to hold energy-related timeseries, specific to portfolios.
"""

class Portfolio:

    def __init__(self, sourced=None, offtake=None):
        pass
    
    @property
    def sourced(self) -> PfLine:
        pass

    @property
    def offtake(self) -> PfLine:
        pass

    @property
    def open(self) -> PfLine:
        pass

    def changefreq(self, freq='MS') -> Portfolio:
        pass