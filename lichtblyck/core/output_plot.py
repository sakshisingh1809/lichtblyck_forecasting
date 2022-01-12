"""
Module with mixins, to add 'plot-functionality' to PfLine and PfState classes.
"""

from __future__ import annotations

import matplotlib
from ..visualize import visualize as vis
from ..tools import nits
from typing import Dict, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:  # needed to avoid circular imports
    from .pfstate import PfState
    from .pfline.abc import PfLine

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
            len(cols),
            1,
            sharex=True,
            sharey=False,
            squeeze=False,
            figsize=(10, len(cols) * 3),
        )

        for col, ax in zip(cols, axes.flatten()):
            color = getattr(vis.Colors.Wqpr, col)
            self.plot_to_ax(ax, col, color=color)
        return fig


class PfStatePlotOutput:
    def plot_to_ax(
        self: PfState, ax: plt.Axes, line: str = "offtake", col: str = None, **kwargs
    ) -> None:
        """Plot a timeseries of a PfState in the portfolio state to a specific axes.

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
        if line == "offtake":
            (-self.offtake).plot_to_ax(ax, col)
            ax.bar_label(
                ax.containers[0], label_type="edge", fmt="%.0f"
            )  # print labels on top of each bar

        elif line == "hedgedfraction":
            hedgefraction = -self.sourced.volume / self.offtake.volume
            vis.plot_timeseries_as_bar(ax, hedgefraction, color="grey")
            ax.bar_label(
                ax.containers[0],
                label_type="edge",
                labels=[f"{val:.0%}" for val in hedgefraction],
            )  # print labels on top of each bar

        elif line == "price":
            vis.plot_timeseries_as_bar(ax, self.unsourcedprice["p"], alpha=0.0)
            ax.bar_label(
                ax.containers[0], label_type="center"
            )  # print labels on top of each bar

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
        self.plot_to_ax(ax1, "offtake", "q")

        # plot Hedged volumne (%)
        ax2 = plt.subplot2grid(shape=(2, 2), loc=(0, 1), colspan=1)

        ax2.set_title("Hedged Fraction")
        ax2.set_ylabel("Percentage")
        ax2.xaxis.set_tick_params(
            labeltop=True
        )  # make x-axis tick labels on the top of a plot
        ax2.xaxis.set_tick_params(labelbottom=False)
        ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        self.plot_to_ax(ax2, "hedgedfraction")

        # plot price
        ax3 = plt.subplot2grid(shape=(2, 2), loc=(1, 0), colspan=1)
        ax3.set_frame_on(False)
        plt.ylim(-1000000, 1000000)
        ax3.set_yticklabels([])  # make yticks disappear
        ax3.get_xaxis().tick_bottom()
        ax3.set_title("Portfolio Price")
        ax3.axes.get_xaxis().set_visible(False)
        plt.yticks(color="w")
        self.plot_to_ax(ax3, "price")


def plot_pfstates(dic: Dict[str, PfState], freq: str = "MS") -> plt.Figure:
    """Plot multiple PfState instances.

    Parameters
    ----------
    dic : Dict[str, PfState]
        Dictionary with PfState instances as values, and their names as the keys.
        The Dictionary argument is a dictionary with pretty portfolio names as keys,
        and the PfState instances as values

    Returns
    -------
    plt.Figure
        The figure object to which the instances were plotted.
    """
    fig, axes = plt.subplots(len(dic) * 2, 2)
    fig.set_size_inches(20, 10)
    pfnames = list(dic.keys())
    pfstates = list(dic.values())
    j = 0

    for i in range(1, len(axes), 2):
        # print(pfnames[j])
        pfs = pfstates[j].changefreq(freq)
        axes[i, 1].axis("off")
        axes[i - 1, 0].set_yticklabels([])

        if i != 1:  # don't remove labels from axes[0,0]
            axes[i - 1, 0].set_xticklabels([])
            axes[i - 1, 1].set_xticklabels([])

        # axes[1, 1].text(0.5, 0.5, "my text")

        pfs.plot_to_ax(axes[i - 1, 0], "offtake", "q")  # plot offtake
        pfs.plot_to_ax(axes[i - 1, 1], "hedgedfraction")  # plot hedgedfraction
        pfs.plot_to_ax(axes[i, 0], "price")  # plot price

        axes[i, 0].set_frame_on(False)
        plt.ylim(-1000000, 1000000)
        axes[i, 0].set_yticklabels([])
        axes[i, 0].get_xaxis().tick_bottom()
        axes[i, 0].axes.get_xaxis().set_visible(False)
        plt.yticks(color="w")
        j = j + 1

    axes[0, 0].set_title("Offtake Volume")
    axes[0, 1].set_title("Hedged Fraction")
    axes[0, 0].xaxis.tick_top()
    axes[0, 1].xaxis.tick_top()

    # rearange the axes for no overlap
    fig.tight_layout()

    # Get the bounding boxes of the axes including text decorations
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(
        list(map(get_bbox, axes.flat)), matplotlib.transforms.Bbox
    ).reshape(axes.shape)

    # Get the minimum and maximum extent, get the coordinate half-way between those
    ymax = (
        np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(axes.shape).max(axis=1)
    )
    ymin = (
        np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(axes.shape).min(axis=1)
    )
    ys = np.c_[ymax[2:-1:2], ymin[1:-2:2]].mean(axis=1)
    ys = [ymax[0], *ys]
    # Draw a horizontal lines at those coordinates
    for y in ys:
        line = plt.Line2D([0, 1], [y, y], transform=fig.transFigure, color="black")
        fig.add_artist(line)

    """
    fig = plt.figure()
    fig.set_size_inches(20, 10)

    for i, (pfname, pfs) in enumerate(dic.items()):
        pfs = pfs.changefreq(freq)
        ax1 = plt.subplot2grid(shape=(len(dic) + 4, 2), loc=(0 + i * 2, 0), colspan=1)
        ax1.xaxis.set_tick_params(
            labeltop=True
        )  # make x-axis tick labels on the top of a plot
        ax1.xaxis.set_tick_params(labelbottom=False)
        ax1.set_title("Offtake Volume")
        pfs.plot_to_ax(ax1, "offtake", "q")

        ax2 = plt.subplot2grid(shape=(len(dic) + 4, 2), loc=(0 + i * 2, 1), colspan=1)
        hedgefraction = -pfs.sourced.volume / pfs.offtake.volume
        ax2.set_title("Hedged Fraction")
        ax2.set_ylabel("Percentage")
        ax2.xaxis.set_tick_params(
            labeltop=True
        )  # make x-axis tick labels on the top of a plot
        ax2.xaxis.set_tick_params(labelbottom=False)
        pfs.plot_to_ax(ax2, "hedgedfraction")

        ax3 = plt.subplot2grid(shape=(len(dic) + 4, 2), loc=(1 + i * 2, 0), colspan=1)
        ax3.set_frame_on(False)
        plt.ylim(-1000000, 1000000)
        ax3.set_yticklabels([])  # make yticks disappear
        ax3.get_xaxis().tick_bottom()
        ax3.axes.get_xaxis().set_visible(False)
        plt.yticks(color="w")
        pfs.plot_to_ax(ax3, "price")
    """
