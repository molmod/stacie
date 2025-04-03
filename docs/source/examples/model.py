#!/usr/bin/env python3

# %% [markdown]
# # Exponential Tail Model Illustration
#
# This notebook creates a plot that shows all the features of the
# [Exponential Tail Model](../theory/autocorrelation_integral/model.md).
# Dimensionless units are used throughout.

# %% [markdown]
# ## Import Libraries and Configure `matplotlib`

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# %%
mpl.rc_file("matplotlibrc")
# %config InlineBackend.figure_formats = ["svg"]

# %% [markdown]
# ## Plot of the Exponential Tail Model
#

# %%
# A frequency axis using RFFT conventions
nfreq = 2000
timestep = 0.2
ks = np.arange(nfreq // 2)
freqs = ks / (timestep * nfreq)

# Model parameters
a_short = 0.3
a_tail = 0.8
corrtime_exp = 2.0

# Derived quantities
r = np.exp(-timestep / corrtime_exp)
cos = np.cos(2 * np.pi * ks / nfreq)
freq_half = np.arccos(2 - np.cosh(timestep / corrtime_exp)) / (2 * np.pi * timestep)
freq_half_approx = 1 / (2 * np.pi * corrtime_exp)

# Model to be plotted
spectrum_model = a_short + a_tail * (1 - r) ** 2 / (1 - 2 * r * cos + r**2)


# %%
def plot_model():
    plt.close("model")
    _, ax = plt.subplots(num="model")
    ax.plot(freqs, spectrum_model, color="k", label="model")
    ax.axhline(a_short + a_tail, alpha=0.5, color="C0", label="a_short + a_tail")
    ax.axhline(
        a_short + 0.5 * a_tail, color="C1", alpha=0.5, label="a_short + a_tail / 2"
    )
    ax.axhline(a_short, color="C2", alpha=0.5, label="a_short")
    ax.axvline(freq_half, alpha=0.5, color="C3", label="f_half")
    ax.axvline(
        freq_half_approx, alpha=0.5, color="C4", ls=":", lw=4, label="f_half_approx"
    )
    ax.set_xlim(0, 4 * freq_half)
    ax.set_ylim(0, (a_short + a_tail) * 1.05)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    ax.legend(loc=0)
    ax.set_title("Exponential Tail Model of the Spectrum")


plot_model()

# %% [markdown]
# ## Plot of the series expansion used to approximate the peak width


# %%
def plot_series():
    fig, ax = plt.subplots()
    us = np.linspace(-1.7627, 1.7627, 501)
    ax.plot(us, np.arccos(2 - np.cosh(us)), label="acos(2-cosh(u))")
    ax.plot(us, abs(us), label="1st order expansion")
    ax.plot(us, abs(us + us**3 / 12), label="3rd order expansion")
    ax.plot(us, abs(us + us**3 / 12 + us**5 / 96), label="5th order expansion")
    ax.set_xlabel("u")
    ax.set_ylabel("f(u)")
    ax.legend(loc=0)


plot_series()
