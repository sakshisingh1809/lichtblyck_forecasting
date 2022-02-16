"""
Module with mixins, to add interactivity-functionality to PfLine and PfState classes (as
a clickable website).
"""
"""
from pyparsing import line
import matplotlib
from ..visualize import visualize as vis
from ..tools import nits
from typing import Dict, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt


class OutputInteractive:
    def plot_to_ax(self: PfState, ax: plt.Axes, line: str ='"offtake', col: str= None, **kwargs) -> None"
    if line():
        pass
"""

from flask import Flask, render_template

app = Flask(__name__)

"""
@app.route("/")
def index():
    return "Test static website!"
"""


@app.route("/")
def index():
    author = "Bob"
    return render_template("index.html", author=author)


if __name__ == "__main__":
    app.run()
