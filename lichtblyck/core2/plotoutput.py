"""
Module with mixins, to add 'plot-functionality' to PfLine and PfState classes.
"""

from __future__ import annotations
from ..visualize import visualize as vis
from ..tools import units
from typing import Dict, TYPE_CHECKING
from matplotlib import pyplot as plt

if TYPE_CHECKING:  # needed to avoid circular imports
    from .pfstate import PfState
    from .pfline import PfLine


class PfLinePlotOutput:
    def plot_to_ax(self: PfLine, ax: plt.Axes, col: str = None, **kwargs) -> None:
        """Plot a timeseries of the PfLine to a specific axes.

        Parameters
        ----------
        ax : plt.Axes
            The axes object to which to plot the timeseries.
        col : str, optional
            The column to plot. Default: plot volume `w` [MW] (if available) or else
            price `p` [Eur/MWh].
        Any additional kwargs are passed to the pd.Series.plot function.
        """
        if not col:
            col = "w" if "w" in self.available else "p"
        how = {"r": "bar", "q": "bar", "p": "step", "w": "line"}.get(col)
        if not how:
            raise ValueError("`col` must be one of {'w', 'q', 'p', 'r'}")
        s = self[col]
        vis.plot_timeseries(ax, s, how=how, **kwargs)

    def plot(self: PfLine, cols: str = "wp") -> plt.Figure:
        """Plot one or more timeseries of the PfLine.

        Parameters
        ----------
        cols : str, optional
            The columns to plot. Default: plot volume `w` [MW] and price `p` [Eur/MWh]
            (if available).

        Returns
        -------
        plt.Figure
            The figure object to which the series was plotted.
        """
        cols = [col for col in cols if col in self.available]
        fig, axes = plt.subplots(
            len(cols), 1, True, False, squeeze=False, figsize=(10, len(cols) * 3)
        )

        for col, ax in zip(cols, axes.flatten()):
            color = getattr(vis.Colors.Wqpr, col)
            unit = units.BU(col)
            self.plot_to_ax(ax, col, color=color)
            ax.set_ylabel(unit)
        return fig


class PfStatePlotOutput:
    def plot_to_ax(
        self: PfState, ax: plt.Axes, line: str = "offtake", col: str = None, **kwargs
    ) -> None:
        """Plot a timeseries of a PfLine in the portfolio state to a specific axes.
        
        Parameters
        ----------
        ax : plt.Axes
            The axes object to which to plot the timeseries.
        line : str, optional
            The pfline to plot. One of {'offtake' (default), 'sourced', 'unsourced', 
            'netposition', 'pnl_costs'}.
        col : str, optional
            The column to plot. Default: plot volume `w` [MW] (if available) or else
            price `p` [Eur/MWh].
        Any additional kwargs are passed to the pd.Series.plot function.
        """
        pass  # TODO

    def plot(self: PfState, cols: str = "wp") -> plt.Figure:
        """Plot one or more timeseries of the portfolio state.
        
        Parameters
        ----------
        cols : str, optional
            The columns to plot. Default: plot volume `w` [MW] and price `p` [Eur/MWh] 
            (if available).

        Returns
        -------
        plt.Figure
            The figure object to which the series was plotted.
        """
        pass  # TODO


def plot_pfstates(dic: Dict[str, PfState]) -> plt.Figure:
    """Plot multiple PfState instances.

    Parameters
    ----------
    dic : Dict[str, PfState]
        Dictionary with PfState instances as values, and their names as the keys.

    Returns
    -------
    plt.Figure
        The figure object to which the instances were plotted.
    """
    pass  # TODO
