"""
Module with mixins, to add 'plot-functionality' to PfLine and PfState classes.
"""

from __future__ import annotations

from ...visualize import visualize as vis
from ...tools import stamps

from typing import Dict, TYPE_CHECKING
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

if TYPE_CHECKING:  # needed to avoid circular imports
    from ..pfstate import PfState
    from ..pfline import PfLine


DEFAULTHOW = {"r": "bar", "q": "bar", "p": "hline", "w": "area", "f": "area"}
DEFAULTFMT = {
    "w": "{:,.1f}",
    "q": "{:,.0f}",
    "p": "{:,.2f}",
    "r": "{:,.0f}",
    "f": "{:.0%}",
}


def defaultkwargs(col, cat):
    kwargs = {}
    kwargs["alpha"] = 0.8
    # Add defaults for each column.
    kwargs["color"] = getattr(vis.Colors.Wqpr, col, "grey")
    kwargs["labelfmt"] = DEFAULTFMT.get(col, "{:.2f}")
    kwargs["how"] = DEFAULTHOW.get(col, "hline")
    kwargs["cat"] = cat
    # Override specific cases.
    if not cat and col == "p":
        kwargs["how"] = "step"
    if cat and col == "f":
        kwargs["how"] = "bar"
    return kwargs


class PfLinePlot:
    def plot_to_ax(
        self: PfLine, ax: plt.Axes, col: str, how: str, labelfmt: str, **kwargs
    ) -> None:
        """Plot a timeseries of the PfLine to a specific axes.

        Parameters
        ----------
        ax : plt.Axes
            The axes object to which to plot the timeseries.
        col : str
            The column to plot. One of {'w', 'q', 'p', 'r'}.
        how : str
            How to plot the data. One of {'jagged', 'bar', 'area', 'step', 'hline'}.
        labelfmt : str
            Labels are added to each datapoint in the specified format. ('' to add no labels)
        Any additional kwargs are passed to the pd.Series.plot function.
        """
        if col not in self.available:
            raise ValueError(
                f"For this PfLine, parameter ``col`` must be one of {', '.join(self.available)}; got {col}."
            )
        vis.plot_timeseries(ax, self[col], how, labelfmt, **kwargs)

    def plot(self: PfLine, cols: str = None) -> plt.Figure:
        """Plot one or more timeseries of the PfLine.

        Parameters
        ----------
        cols : str, optional
            The columns to plot. Default: plot volume (in [MW] for daily values and
            shorter, [MWh] for monthly values and longer) and price `p` [Eur/MWh]
            (if available).

        Returns
        -------
        plt.Figure
            The figure object to which the series was plotted.
        """
        # Plot on category axis if freq monthly or longer, else on time axis.
        category = stamps.freq_shortest(self.index.freq, "MS") == "MS"

        # If columns are specified, plot these. Else: take defaults, based on what's available
        if cols is None:
            cols = ""
            if "q" in self.available:
                cols += "q" if category else "w"
            if "p" in self.available:
                cols += "p"
        else:
            cols = [col for col in cols if col in self.available]
            if not cols:
                raise ValueError("No columns to plot.")

        # Create the plots.
        size = (10, len(cols) * 3)
        fig, axes = plt.subplots(
            len(cols), 1, sharex=True, sharey=False, squeeze=False, figsize=size
        )

        for col, ax in zip(cols, axes.flatten()):
            kwargs = defaultkwargs(col, category)
            s = getattr(self, col)
            vis.plot_timeseries(ax, s, **kwargs)

        return fig


class PfStatePlot:
    # def plot_to_ax(
    #     self: PfState, ax: plt.Axes, line: str = "offtake", col: str = None, **kwargs
    # ) -> None:
    #     """Plot a timeseries of a PfState in the portfolio state to a specific axes.

    #     Parameters
    #     ----------
    #     ax : plt.Axes
    #         The axes object to which to plot the timeseries.
    #     line : str, optional
    #         The pfline to plot. One of {'offtake' (default), 'sourced', 'unsourced',
    #         'netposition', 'procurement', 'sourcedfraction'}.
    #     col : str, optional
    #         The column to plot. Default: plot volume `w` [MW] (if available) or else
    #         price `p` [Eur/MWh].
    #     Any additional kwargs are passed to the pd.Series.plot function.
    #     """
    #     if line == "offtake":
    #         how = DEFAULTHOW.get(col, "step")
    #         (-self.offtake).plot_to_ax(ax, col, how)
    #         ax.bar_label(
    #             ax.containers[0], label_type="edge", fmt="%,.0f".replace(",", " ")
    #         )

    #     elif line.endswith("sourcedfraction"):  # (un)sourcedfraction
    #         fractions = getattr(self, line)
    #         vis.plot_timeseries(ax, fractions, how="bar", color="grey")
    #         ax.bar_label(
    #             ax.containers[0],
    #             label_type="edge",
    #             labels=fractions.apply("{:.0%}".format),
    #         )  # print labels on top of each bar

    #     elif line == "sourced":
    #         self.sourced.plot_to_ax(
    #             ax,
    #             col,
    #         )
    #         if col == "p":

    #             vis.plot_timeseries(ax, self.unsourcedprice["p"], how="bar", alpha=0.0)
    #             ax.bar_label(
    #                 ax.containers[0], label_type="center", fmt="%.2f"
    #             )  # print labels on top of each bar

    def plot(self: PfState) -> plt.Figure:
        """Plot one or more timeseries of the portfolio state.

        Parameters
        ----------
        None

        Returns
        -------
        plt.Figure
            The figure object to which the series was plotted.
        """
        gridspec = {"width_ratios": [1, 1], "height_ratios": [4, 1]}
        fig, axes = plt.subplots(
            2, 2, sharex=True, gridspec_kw=gridspec, figsize=(10, 6)
        )
        axes = axes.flatten()

        # If freq is MS or longer: use categorical axes. Plot volumes in MWh.
        # If freq is D or shorter: use time axes. Plot volumes in MW.
        category = stamps.freq_shortest(self.index.freq, "MS") == "MS"

        # Volumes.
        if category:
            s, kwargs = -1 * self.offtake.q, defaultkwargs("q", category)
        else:
            s, kwargs = -1 * self.offtake.w, defaultkwargs("w", category)
        vis.plot_timeseries(axes[0], s, **kwargs)

        # Sourced fraction.
        vis.plot_timeseries(
            axes[1], self.sourcedfraction, **defaultkwargs("f", category)
        )

        # Procurement Price.
        vis.plot_timeseries(axes[2], self.pnl_cost.p, **defaultkwargs("p", category))

        # Empty.
        axes[3].axis("off")

        # Set titles.
        axes[0].set_title("Offtake")
        axes[1].set_title("Sourced fraction")
        axes[2].set_title("Procurement price")

        # Format tick labels.
        axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        axes[0].yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda x, p: "{:,.0f}".format(x).replace(",", " ")
            )
        )

        # Set ticks.
        axes[0].xaxis.set_tick_params(labeltop=False, labelbottom=True)
        axes[1].xaxis.set_tick_params(labeltop=False, labelbottom=True)
        axes[2].xaxis.set_tick_params(labeltop=False, labelbottom=False)

        fig.tight_layout()
        return fig


def plot_pfstates(dic: Dict[str, PfState], freq: str = "MS") -> plt.Figure:
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

    gridspec = {"width_ratios": [0.3, 1, 1], "height_ratios": [4, 1] * len(dic)}
    figsize = (15, 5 * len(dic))
    fig, axes = plt.subplots(len(dic) * 2, 3, gridspec_kw=gridspec, figsize=figsize)
    axesgroups = axes.flatten().reshape((len(dic), 6))

    # Share x axes.
    sharex = axesgroups[:, (1, 2, 4)].flatten()
    for ax1, ax2 in zip(sharex[1:], sharex[:-1]):
        ax1.sharex(ax2)
    # Share y axes.
    sharey = axesgroups[:, 2]
    for ax1, ax2 in zip(sharey[1:], sharey[:-1]):
        ax1.sharey(ax2)

    # TODO: resample all to have same index (frequency and length).

    for i, ((pfname, pfs), axes) in enumerate(zip(dic.items(), axesgroups)):

        # If freq is MS or longer: use categorical axes. Plot volumes in MWh.
        # If freq is D or shorter: use time axes. Plot volumes in MW.
        category = stamps.freq_shortest(pfs.index.freq, "MS") == "MS"

        # Portfolio name.
        axes[0].text(
            0,
            1,
            pfname.replace(" ", "\n"),
            fontsize=12,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="left",
        )
        axes[0].axis("off")

        # Volumes.
        if category:
            s, kwargs = -1 * pfs.offtake.q, defaultkwargs("q", category)
        else:
            s, kwargs = -1 * pfs.offtake.w, defaultkwargs("w", category)
        vis.plot_timeseries(axes[1], s, **kwargs)

        # Sourced fraction.
        vis.plot_timeseries(
            axes[2], pfs.sourcedfraction, **defaultkwargs("f", category)
        )

        # Empty.
        axes[3].axis("off")

        # Procurement Price.
        vis.plot_timeseries(axes[4], pfs.pnl_cost.p, **defaultkwargs("p", category))

        # Empty.
        axes[5].axis("off")

        # Tick formatting.
        axes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        axes[1].yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda x, p: "{:,.0f}".format(x).replace(",", " ")
            )
        )

        for a, ax in enumerate(axes):
            if i == 0 and a in [1, 2]:
                ax.xaxis.set_tick_params(labelbottom=False, labeltop=True, pad=25)
            else:
                ax.xaxis.set_tick_params(labelbottom=False, labeltop=False)

        if i == 0:
            axes[1].set_title("Offtake Volume &\nprocurement price", y=1.27)
            axes[2].set_title("Sourced fraction", y=1.27)

    return
    draw_horizontal_lines(fig, axes)  # draw horizontal lines between portfolios


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
