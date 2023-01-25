"""
Consumption in each climate zone.
"""

from pathlib import Path
import pandas as pd

SOURCE = Path(__file__).parent / "sourcedata" / "weights.xlsx"


def weights() -> pd.DataFrame:
    """
    Weights of the climate zones.

    Parameters
    ----------
    None

    Returns
    -------
    pd.DataFrame
        with climate zones as index, customer segment as columns.
    """
    df = pd.read_excel(io=SOURCE)
    df = df.set_index("zone")
    df.index = df.index.map(lambda z: f"t_{z}")
    return df
