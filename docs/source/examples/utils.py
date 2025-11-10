# %% [markdown]
# # Utility Module for Plots Reused in Multiple Examples.

# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2


__all__ = (
    "plot_instantaneous_percentiles",
    "plot_cumulative_temperature_histogram",
    "compute_msds",
)


def compute_msds(series: NDArray[float], lags: NDArray[int]) -> NDArray[float]:
    """Compute the mean squared displacements of one or more time series.

    Parameters
    ----------
    series
        The time series to compute the MSDs for, with shape `(*prefix, nstep)`.
        It is assumed that the last index corresponds to time.
    lags
        The lag times to consider (in number of time steps), with shape `(nlag,)`.

    Returns
    -------
    NDArray[float]
        The mean squared displacements for the specified lag times,
        with shape `(nlag,)`.
    """
    series = np.asarray(series)
    lags = np.asarray(lags)
    if lags.ndim != 1:
        raise ValueError("lags must be a 1D array")
    msds = np.zeros(lags.shape)
    for i, lag in enumerate(lags):
        diffs = np.diff(series[..., ::lag], axis=-1)
        msds[i] = np.mean(diffs**2)
    return msds


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


def plot_cumulative_temperature_histogram(
    ax: plt.Axes,
    temps: NDArray[float],
    temp_d: float,
    ndof: int,
    temp_unit_str: str,
    nbin: int = 100,
):
    """Plot a cumulative histogram of the temperature.

    Parameters
    ----------
    ax
        The axes to plot on.
    temps
        The temperature data to plot.
        This is expected to be a 2D array with shape (ntraj, nstep).
        Cumulative histograms of individual trajectories will be plotted,
        together with the combined and theoretical cumulative histogram.
    temp_d
        The desired temperature for the theoretical cumulative histogram.
    ndof
        The number of degrees of freedom for the system.
    temp_unit_str
        A string representing the unit of temperature.
    nbin
        The number of bins for the histogram.
    """
    label = "Individual NVE"
    quantiles = (np.arange(nbin) + 0.5) / nbin
    for temp in temps:
        temp.sort()
        ax.plot(
            np.quantile(temp, quantiles),
            quantiles,
            alpha=0.2,
            color="C0",
            label=label,
        )
        label = "__nolegend__"
    ax.plot(
        np.quantile(temps, quantiles),
        quantiles,
        color="black",
        label="Combined NVE",
    )
    temp_axis = np.linspace(np.min(temps), np.max(temps), 100)
    ax.plot(
        temp_axis,
        chi2.cdf(temp_axis * ndof / temp_d, ndof),
        color="C3",
        ls=":",
        lw=4,
        label="NVT exact",
    )
    ax.legend()
    ax.set_title("Cumulative Distribution of the Instantaneous Temperature")
    ax.set_xlabel(f"Temperature [{temp_unit_str}]")
    ax.set_ylabel("Cumulative Probability")
