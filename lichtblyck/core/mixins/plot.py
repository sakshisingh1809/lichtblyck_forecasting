"""
Module with mixins, to add 'plot-functionality' to PfLine and PfState classes.
"""

from __future__ import annotations
from tkinter import font

from ...visualize import visualize as vis
from ...tools import nits
from typing import Dict, TYPE_CHECKING
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

if TYPE_CHECKING:  # needed to avoid circular imports
    from ..pfstate import PfState
    from ..pfline import PfLine

DEFAULTPLOTTYPES = {"r": "bar", "q": "bar", "p": "step", "w": "line"}


class PfLinePlot:
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
            # gridspec_kw={"height_ratios": [4, 1]},
        )

        for col, ax in zip(cols, axes.flatten()):
            color = getattr(vis.Colors.Wqpr, col)
            self.plot_to_ax(ax, col, color=color)
        return fig


class PfStatePlot:
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
                ax.containers[0], label_type="edge", fmt="%.0f".replace(",", " ")
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
                ax.containers[0], label_type="center", fmt="%.2f"
            )  # print labels on top of each bar

    def plot(self: PfState, cols: str = "wp", freq: str = "MS") -> plt.Figure:
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
            2, 3, gridspec_kw={"width_ratios": [0.3, 2, 2], "height_ratios": [4, 1],},
        )

        fig.set_size_inches(20, 10)
        pf = self.changefreq(freq)

        axes[0, 0].axis("off")
        axes[1, 0].axis("off")
        axes[1, 2].axis("off")
        axes[0, 1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

        # make x-axis tick labels on the top of a plot
        axes[0, 1].xaxis.set_tick_params(labeltop=True)
        axes[0, 2].xaxis.set_tick_params(labeltop=True)
        axes[0, 1].xaxis.set_tick_params(labelbottom=False)
        axes[0, 2].xaxis.set_tick_params(labelbottom=False)

        pf.plot_to_ax(axes[0, 1], "offtake", "q")  # plot offtake
        pf.plot_to_ax(axes[0, 2], "hedgedfraction")  # plot hedgedfraction
        pf.plot_to_ax(axes[1, 1], "price")  # plot price

        # print offtake units
        print_labels(axes, 0, 0, axes[0, 1].get_ylabel())

        # print price units
        print_labels(axes, 1, 0, axes[1, 1].get_ylabel())

        plt.ylim(-1000000, 1000000)
        axes[1, 1].set_frame_on(False)
        axes[0, 1].axes.get_yaxis().set_visible(False)
        axes[0, 2].axes.get_yaxis().set_visible(False)
        axes[1, 1].axes.get_yaxis().set_visible(False)
        axes[1, 1].axes.get_xaxis().set_visible(False)
        plt.yticks(color="w")


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

    ratios = []
    for i in range(len(dic)):
        ratios.append([4, 1])
    ratios_list = [item for sublist in ratios for item in sublist]

    fig, axes = plt.subplots(
        len(dic) * 2,
        4,
        gridspec_kw={"width_ratios": [0.3, 0.3, 2, 2], "height_ratios": ratios_list},
    )

    fig.set_size_inches(20, 10)
    pfnames = list(dic.keys())
    pfstates = list(dic.values())
    j = 0

    """ EXTRA: Check for intersection for all the pfstates before proceeding inside the loop.
    We need to check for intersection first by taking pfstate1.offtake.index.INTERSECTION(pfstate2.offtake.index)
    After this we extract and only work with the common part of all the pfstates."""

    for i in range(1, len(axes), 2):
        pfs = pfstates[j].changefreq(freq)

        axes[i - 1, 0].axis("off")
        axes[i, 0].axis("off")
        axes[i - 1, 1].axis("off")
        axes[i, 1].axis("off")
        axes[i, 3].axis("off")

        if i != 1:  # don't remove labels from axes[0,0]
            axes[i - 1, 2].set_xticklabels([])
            axes[i - 1, 3].set_xticklabels([])

        axes[i - 1, 3].xaxis.set_tick_params(
            labeltop=True
        )  # make x-axis tick labels on the top of a plot
        axes[i - 1, 3].xaxis.set_tick_params(labelbottom=False)
        axes[i - 1, 3].yaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(1.0)
        )

        pfs.plot_to_ax(axes[i - 1, 2], "offtake", "q")  # plot offtake
        pfs.plot_to_ax(axes[i - 1, 3], "hedgedfraction")  # plot hedgedfraction
        pfs.plot_to_ax(axes[i, 2], "price")  # plot price

        # print portfolio names on the left most (i-1,0), eg. (0,0), (2,0),...
        print_labels(axes, i - 1, 0, pfnames[j])

        # print offtake units on next column (i-1,1), eg. (0,1), (2,1),...
        print_labels(axes, i - 1, 1, axes[i - 1, 2].get_ylabel())

        # print price units on next column (i,1), eg. (0,1), (2,1),...
        print_labels(axes, i, 1, axes[i, 2].get_ylabel())

        plt.ylim(-1000000, 1000000)
        axes[i, 2].set_frame_on(False)
        axes[i - 1, 2].axes.get_yaxis().set_visible(False)
        axes[i - 1, 3].axes.get_yaxis().set_visible(False)
        axes[i, 2].axes.get_yaxis().set_visible(False)

        axes[i - 1, 2].set_yticklabels([])
        axes[i, 2].set_yticklabels([])
        axes[i, 2].get_xaxis().tick_bottom()
        axes[i, 2].axes.get_xaxis().set_visible(False)
        plt.yticks(color="w")
        j = j + 1

    axes[0, 2].set_title("Offtake Volume", y=1.3)
    axes[0, 3].set_title("Hedged Fraction [%]", y=1.3)
    axes[0, 2].xaxis.tick_top()
    axes[0, 3].xaxis.tick_top()

    draw_horizontal_lines(fig, axes)  # draw horizontal lines between portfolios


def print_labels(axes, x, y, value):
    axes[x, y].text(
        0.5, 0.5, value, fontsize=14, fontweight="bold", horizontalalignment="center",
    )
    return


def draw_horizontal_lines(fig, axes):
    """Function to draw horizontal lines between multiple portfolios.
    This function does not return anything, but tries to plot a 2D line after every 2 axes, eg.
    after (0,2), (0,4),... beacuse each portfolio requires 2x4 axes in the fig (where rows=2, columns=4).

    Parameters
    ----------
    fig : plt.subplots()
    axes : plt.subplots()
    """
    # rearange the axes for no overlap
    fig.tight_layout()

    # Get the bounding boxes of the axes including text decorations
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(
        list(map(get_bbox, axes.flat)), matplotlib.transforms.Bbox
    ).reshape(axes.shape)

    """TO CORRECT: the horizontal line is not exactly in the middle of two graphs.
    It is more inclined towards the second or next graph in the queue.
    Each pftstate has 4x4 grid and this is plotted in the same graph, but as subgraphs.
    """

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
