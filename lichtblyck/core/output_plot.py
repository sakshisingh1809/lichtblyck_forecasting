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
            'netposition', 'pnl_costs', 'hedgedfraction'}.
        col : str, optional
            The column to plot. Default: plot volume `w` [MW] (if available) or else
            price `p` [Eur/MWh].
        Any additional kwargs are passed to the pd.Series.plot function.
        """
        if not col:
            col = "w" if "w" in self.available else "p"
        how = {"r": "bar", "q": "bar", "p": "step", "w": "line"}.get(col)
        if not how:
            raise ValueError(f"`col` must be one of {', '.join(self.available)}.")
        s = self[col]
        vis.plot_timeseries(ax, s, how=how, **kwargs)

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

        fig = plt.figure()
        fig.set_size_inches(20, 10)

        # plot Offtake
        ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), colspan=1)
        ax1.xaxis.set_tick_params(
            labeltop=True
        )  # make x-axis tick labels on the top of a plot
        ax1.xaxis.set_tick_params(labelbottom=False)
        ax1.set_title("Offtake Volume")
        (-self.offtake).plot_to_ax(ax1, "q")

        ax1.bar_label(
            ax1.containers[0], label_type="edge", fmt="%.0f"
        )  # print labels on top of each bar

        # plot Hedged volumne (%)
        ax2 = plt.subplot2grid(shape=(2, 2), loc=(0, 1), colspan=1)
        hedgefraction = -self.sourced.volume / self.offtake.volume
        ax2.set_title("Hedged Fraction")
        ax2.set_ylabel("Percentage")
        ax2.xaxis.set_tick_params(
            labeltop=True
        )  # make x-axis tick labels on the top of a plot
        ax2.xaxis.set_tick_params(labelbottom=False)
        vis.plot_timeseries_as_bar(ax2, hedgefraction * 100, color="grey")
        ax2.bar_label(
            ax2.containers[0], label_type="edge", fmt="%.0f",
        )  # print labels on top of each bar

        # plot price
        ax3 = plt.subplot2grid(shape=(2, 2), loc=(1, 0), colspan=1)
        ax3.set_frame_on(False)
        plt.ylim(-1000000, 1000000)
        ax3.set_yticklabels([])  # make yticks disappear
        ax3.get_xaxis().tick_bottom()
        ax3.set_title("Portfolio Price")
        ax3.axes.get_xaxis().set_visible(False)
        plt.yticks(color="w")
        vis.plot_timeseries_as_bar(ax3, self.unsourcedprice["p"], alpha=0.0)
        ax3.bar_label(
            ax3.containers[0], label_type="center"
        )  # print labels on top of each bar


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

    # plot for offtake [GWh]
    if Dict.keys() == "offtakevolume":
        ax.set_xlabel("GWh")
        ax.set_title("Offtake [GWh]")
        ax.bar(Dict.values(), color=vis.Colors.Wqpr)
        for index, data in enumerate(Dict.values()):
            plt.text(x=index, y=data + 1, s=f"{data}", fontdict=dict(fontsize=20))

    elif Dict.keys() == "sourced":
        ax.set_title("Hedged [%]")

    elif Dict.keys() == "unsourcedprice":
        ax.set_xlabel("Euro/MWh")
        ax.set_title("Unhedged")

    plt.tight_layout()
    plt.show()
