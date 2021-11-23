"""Module with tools for dealing with units ("'nits" to keep "units" available in 
name space.) """

from pathlib import Path
from typing import Union, Dict
import pint
import pint_pandas
import pandas as pd


path = Path(__file__).parent / "unitdefinitions.txt"

ureg = pint_pandas.PintType.ureg = pint.UnitRegistry(
    str(path), system="powerbase", auto_reduce_dimensions=True, case_sensitive=False,
)
ureg.default_format = "~P"  # short by default
ureg.setup_matplotlib()

# Set for export.
PA_ = pint_pandas.PintArray
Q_ = ureg.Quantity


# def to_pref_unit(self: pint.Quantity):
#     for unit in (ureg.MW, ureg.euro_per_MWh):
#         if self.dimensionality == unit.dimensionality:
#             return self.to(unit)
#     return self

# def cast2quant(val, unit:pint.Unit) -> pint.Quantity:
#     """Cast a value `val` to a quantity with the given unit."""
#     return Q_(val, unit)

NAMES_AND_UNITS = {
        "w": ureg.MW,
        "q": ureg.MWh,
        "p": ureg.euro_per_MWh,
        "r": ureg.euro,
        "duration": ureg.hour,
        "t": ureg.degC,
    }


def unit2name(unit: pint.Unit) -> str:
    """Find the standard column name belonging to unit `unit`. Checks on dimensionality, not exact unit."""
    for name, u in NAMES_AND_UNITS.items():
        if u.dimensionality == unit.dimensionality:
            return name
    return ValueError(f"No standard name found for unit '{unit}'.")

def name2unit(name: str) -> pint.Unit:
    """Find standard unit belonging to a column name."""
    if name in NAMES_AND_UNITS:
        return NAMES_AND_UNITS[name]
    raise ValueError(f"No standard unit found for name '{name}'.")


def set_unit(s: pd.Series, unit: Union[pint.Unit, str]) -> pd.Series:
    """Make series unit-aware. If series is already unit-aware, convert to specified unit.
    If not, assume values are in specified unit.

    Parameters
    ----------
    s : pd.Series
    unit : Union[pint.Unit, str]

    Returns
    -------
    pd.Series
        Same as input series, but with specified unit.
    """
    if not isinstance(unit, pint.Unit):
        unit = ureg.Unit(unit)
    dtype = f"pint[{unit}]"
    return s.astype(dtype)  # sets unit if none set yet, otherwise converts if possible


def set_units(
    df: pd.DataFrame, units: Dict[str, Union[pint.Unit, str]]
) -> pd.DataFrame:
    """Make dataframe unit-aware. If dataframe is already unit-aware, convert to specified 
    units. If not, assume values are in specified unit.

    Parameters
    ----------
    df : pd.DataFrame
    units : Dict[str, Union[pint.Unit, str]]
        key = column name, value = unit to set to that column

    Returns
    -------
    pd.DataFrame
        Same as input dataframe, but with specified units.
    """
    df = df.copy()  # don't change dataframe
    for name, unit in units.items():
        df[name] = set_unit(df[name], unit)
    return df


