# %% [markdown]
# # Correlation Time Analysis of Cloud Cover Data
#
# This example is inspired by the work of P. A. Jones {cite:p}`jones_1992_cloudcover`
# on the analysis of temporal (and spatial) correlations in cloud cover data.
# Jones observed an exponential decay of the autocorrelation function,
# with half-life times ranging from 5 to 40 hours.
#
# Here, we analyze a time series of cloud cover data obtained from
# the [Open-Meteo](https://open-meteo.com/) platform.
# The data correspond to hourly cloud cover observations
# (in percentage of the sky covered by clouds) in Ghent, Belgium,
# from January 1, 2010 to January 1, 2020.

# %% [markdown]
# ## Library Imports and Matplotlib Configuration

# %%
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from path import Path
from stacie import (
    compute_spectrum,
    estimate_acint,
    LorentzModel,
    UnitConfig,
    plot_extras,
    plot_fitted_spectrum,
)
from stacie.utils import split

# %%
mpl.rc_file("matplotlibrc")
# %config InlineBackend.figure_formats = ["svg"]

# %% [markdown]
# ## Cloud Cover Data
#
# In this example, cloud cover is expressed as the fraction of the sky covered by clouds,
# ranging from 0 (clear sky) to 1 (completely overcast).

# %%
# You normally do not need to change this path.
# It only needs to be overridden when building the documentation.
DATA_ROOT = Path(os.getenv("DATA_ROOT", "./")) / "cloud-cover/"

# Load data and convert to a fraction from 0 to 1.
cover = (
    np.loadtxt(
        DATA_ROOT / "cloud-cover-ghent-2010-2020.csv",
        delimiter=",",
        usecols=1,
        skiprows=4,
    )
    / 100
)
print(f"Number of measurements: {cover.shape}")

# %% [markdown]
# ## Sample of the Cloud Cover Data
#
# To get a first impression of the data, we plot a small sample.


# %%
def plot_sample():
    plt.close("sample")
    _, ax = plt.subplots(num="sample")
    time = np.arange(240)
    ax.plot(time, cover[:240], "C0-")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Cloud Cover Fraction")
    ax.set_title("Sample of Cloud Cover Data (Ghent, First 10 Days of 2010)")
    ax.set_ylim(-0.05, 1.05)


plot_sample()

# %% [markdown]
# ## Histogram of Cloud Cover Data
#
# The histogram below shows the distribution of cloud cover values.


# %%
def plot_histogram():
    plt.close("histogram")
    _, ax = plt.subplots(num="histogram")
    ax.hist(cover, bins=np.linspace(0, 1, 21), density=True, ec="w")
    ax.set_xlabel("Cloud Cover Fraction")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(0, 1)
    ax.set_title("Histogram of Cloud Cover Data (Ghent, 2010-2020)")


plot_histogram()

# %% [markdown]
# This histogram reflects the typical weather pattern in Belgium,
# with plenty of cloudy days.
# This is also reflected in the normalized standard deviation (NS),
# as defined by Jones {cite:p}`jones_1992_cloudcover`:

# %%
cc_mean = np.mean(cover)
cc_std = np.std(cover)
cc_ns = cc_std / np.sqrt(cc_mean * (1 - cc_mean))
print(f"    Mean cloud cover: {cc_mean:.3f}")
print(f"  Standard deviation: {cc_std:.3f}")
print(f"Normalized std. dev.: {cc_ns:.3f}")

# %% [markdown]
# Compared to the values in Figure 10 of Jones's work,
# this is a relatively high NS, which has been associated with
# longer correlation times.

# %% [markdown]
# ## Autocorrelation Function
#
# Before applying STACIE to the data, let's first compute and plot the
# autocorrelation function (ACF) directly.
#
# Mind the normalization: due to the zero padding in `np.correlate`,
# the number of terms contributing to the ACF decreases with lag time.
# Furthermore, the ACF is not normalized to 1 in this plot,
# as we want to keep the amplitude information.


# %%
AMP_EXP = 0.0347  # From STACIE Lorentz B parameter, see below.
TAU_EXP = 57.6  # From STACIE corrtime_exp, see below.


def plot_acf():
    plt.close("acf")
    _, ax = plt.subplots(num="acf")
    delta = cover - np.mean(cover)
    acf = np.correlate(delta, delta, mode="same")
    nkeep = acf.size // 2
    acf = acf[nkeep:] / (len(acf) - np.arange(nkeep))
    time = np.arange(240)
    ax.plot(time, acf[:240], "C0-")
    ax.plot(
        time,
        AMP_EXP * np.exp(-time / TAU_EXP),
        "C1--",
        label=r"exp(-t/\tau_\mathrm{exp})",
    )
    xticks = np.arange(0, 241, 24)
    ax.set_xlim(-1, 240)
    ax.set_xticks(xticks)
    ax.grid(True, axis="x")
    ax.set_xlabel("Lag Time [h]")
    ax.set_ylabel("Autocorrelation Function")
    ax.set_title("Autocorrelation Function of Cloud Cover Data (Ghent, 2010-2020)")


plot_acf()

# %% [markdown]
# The ripples in the ACF are due to diurnal cycles,
# resulting in weak correlations at multiples of 24 hours.
# Superimposed on these diurnal effects,
# an overall exponential decay of the ACF can be observed.
# However, it is difficult to fit an exponential function to the ACF
# due to (i) the ripples and (ii) non-exponential short-time effects.
# Hence, fitting an exponential function can at best provide
# a rough estimate of the correlation time.

# Below, it is shown how to use STACIE instead to perform a more robust analysis.
# The exponential decay shown in the plot is derived from STACIE's output below.
# It is only expected to be representative at long lag times.

# %% [markdown]
# ## Autocorrelation Time
#
# The following cells perform a standard time correlation analysis using STACIE.
# The `prefactors` argument is set so that the resulting autocorrelation integral
# is the [variance of the mean](../properties/error_estimates.md) cloud cover.
# To reduce the noise of the spectrum, the data is split into 20 blocks,
# and the resulting spectra are averaged.

# %%
# Compute spectrum
spectrum = compute_spectrum(
    split(cover, 20),
    include_zero_freq=False,
    prefactors=2.0 / len(cover),
)

# Estimate autocorrelation time
uc = UnitConfig(
    time_unit_str="h",
    freq_unit_str="1/h",
    time_fmt=".1f",
    freq_fmt=".3f",
)
result = estimate_acint(spectrum, LorentzModel(), verbose=True, uc=uc)

# %% [markdown]
# As expected, the correlation times are on the order of one or a few days.
# The exponential correlation time, which is about 60 hours,
# is the longest because it captures the slowest relaxation process.
# The integrated correlation time is notably shorter at about 20 hours.
#
# In the code below, we also derive the $B$ parameter of the Lorentz model,
# which has been used to plot the exponential decay in the ACF above.

# %%
pars = result.props["pars"]
tau_exp = result.corrtime_exp
amp_exp = (pars[0] - pars[1] / pars[2]) / (2 * tau_exp)
print(f"TAU_EXP = {tau_exp:.4f} h")
print(f"AMP_EXP = {amp_exp:.4f}")

# %% [markdown]
# The plots below show the fitted spectrum and additional diagnostics.
# These plots can be used to evaluate the quality of the fit
# and confirm that the Lorentz model is appropriate in this case.

# %%
plt.close("fitted")
_, ax = plt.subplots(num="fitted")
plot_fitted_spectrum(ax, uc, result)

# %%
plt.close("extras")
_, axs = plt.subplots(2, 2, num="extras")
plot_extras(axs, uc, result)

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(result.acint - 5.1612) > 5e-3:
    raise ValueError(f"Wrong acint: {result.acint:.4e}")
if abs(result.corrtime_exp - 57.590) > 5e-2:
    raise ValueError(f"Wrong corrtime_exp: {result.corrtime_exp:.4e}")
