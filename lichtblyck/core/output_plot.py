"""
Module with mixins, to add 'plot-functionality' to PfLine and PfState classes.
"""

from __future__ import annotations
from ..visualize import visualize as vis
from ..tools import nits
from typing import Dict, TYPE_CHECKING
from matplotlib import pyplot as plt

if TYPE_CHECKING:  # needed to avoid circular imports
    from .pfstate import PfState
    from .pfline import PfLine

DEFAULTPLOTTYPES = {"r": "bar", "q": "bar", "p": "step", "w": "line"}


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
        how = DEFAULTPLOTTYPES.get(col)
        if not how:
            raise ValueError(f"`col` must be one of {', '.join(self.available)}.")
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
            self.plot_to_ax(ax, col, color=color)
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
        fig, axes = plt.subplots(
            1, 2, sharex=True, sharey=True, squeeze=True, figsize=(20, 10)
        )
        (-self.offtake).plot_to_ax(axes[0], 'q')
        self.sourced.plot_to_ax(axes[1], 'q', color='grey')
        axes[0].set_title('Offtake')
        axes[1].set_title('Sourced')
        # hedgefraction = self.sourced.volume / -self.offtake.volume
        # vis.plot_timeseries_as_bar(axes[1], hedgefraction, color='grey')


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

    fig, ax = plt.subplots(
        len(Dict), 1, True, False, squeeze=False, figsize=(10, len(Dict) * 3)
    )
    # make x-axis tick labels on the top of a plot
    plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.labelbottom"] = False
    plt.rcParams["xtick.top"] = plt.rcParams["xtick.labeltop"] = True

    # plot for offtake [MWh]
    if Dict.keys() == "offtake":
        ax.set_xlabel("MWh")
        ax.set_title("Offtake [MWh]")
        bar_plot = ax.bar(Dict.values(), color=vis.Colors.Wqpr)
        for index, data in enumerate(y):
            plt.text(x=index, y=data + 1, s=f"{data}", fontdict=dict(fontsize=20))

    plt.show()
