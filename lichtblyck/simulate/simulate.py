
import numpy as np
from scipy.stats import norm

def randomwalkfactor(vola: float, time: float, percentile: float = None) -> float:
    """Calculate the factor with which to multiply a variable undergoing a random walk.

    Parameters
    ----------
    vola : float
        Volatility in %/a.
    time : float
        Time step in a.
    percentile : float. Optional. Default: random value between 0 and 1.
        Percentile for which to get the factor.

    Returns
    -------
    float
        Multiplication factor
    """

    if vola == 0:
        vola = 1e-5 
    if percentile is None:
        percentile = np.random.rand()
    mu = -0.5 * (vola ** 2)
    loc, scale = mu * time, vola * np.sqrt(time)
    return np.exp(norm(loc, scale).ppf(percentile))
