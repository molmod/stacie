# %% [markdown]
# # Utility module for plots reused in several examples.

# %%
import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt


def plot_instantaneous_percentiles(
    ax: plt.Axes,
    time: NDArray[float],
    data: NDArray[float],
    percents: ArrayLike,
    expected: ArrayLike | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
):
    """Plot time-dependent percentiles of a data set.

    Parameters
    ----------
    ax
        The axes to plot on.
    time
        The time points corresponding to the data.
    data
        The data to plot. It should be a 2D array with shape (nsample, nstep).
    percents
        The percentages for which to plot the percentiles.
    expected
        The expected values to plot as horizontal lines.
    ymin
        Y-axis lower limit.
    ymax
        Y-axis upper limit.
    """
    for percent, percentile in zip(percents, np.percentile(data, percents, axis=0)):
        ax.plot(time, percentile, label=f"{percent} %")
    if expected is not None:
        for value in expected:
            ax.axhline(value, color="black", linestyle=":")
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_title("Percentiles during the equilibration run")
    ax.legend()
