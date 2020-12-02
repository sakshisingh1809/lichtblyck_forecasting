"""
Module to analyse scenario data
"""

from scipy.stats import norm


def expected_shortfall(loc:float=0, scale:float=1, *, quantile:float=None, x0:float=None) -> float:
    """
    Calculate the expected loss, for a normal distribution with mean 'loc' 
    and standard deviation 'scale'.
    
    It is assumed, that a value 'x0', corresponding to quantile 'quantile', is 
    priced in. This function calculates, for the cases that the costs exceed 
    x0, how much the average excess cost (i.e., the expected shortfall) is.
        
    Exactly one of 'quantile' and 'x0' must be specified. The right-hand tail
    is evaluated.
    """
    if sum((x0 is not None, quantile is not None)) != 1:
        raise ValueError("Of parameters 'quantile' and 'bound', exactly one must be specified.")
    if quantile is None:
        quantile = norm(loc, scale).cdf(x0)
    if x0 is None:
        x0 = norm(loc, scale).ppf(quantile)
    
    expected_cost = loc + scale * norm.pdf(norm.ppf(quantile)) / (1 - quantile)
    return expected_cost - x0

