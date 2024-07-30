#!/usr/bin/env python

# %% [markdown]
# # Applicability of the Exponential Tail Model
#
# Stacie's Exponential Tail Model assumes that
# the autocorrelation function decays exponentially for large lag times.
# Not all dynamical systems exhibit this exponential relaxation.
# If you still want to apply Stacie to such cases,
# you need to implement an appropriate model for the low-frequency spectrum.
#
# This notebook applies Stacie to numerical solutions of
# [Thomas' Cyclically Symmetric Attractor](https://en.wikipedia.org/wiki/Thomas%27_cyclically_symmetric_attractor):
#
# $$
#   \frac{\mathrm{d}x}{\mathrm{d}t} &= \sin(y) - bx
#   \\
#   \frac{\mathrm{d}y}{\mathrm{d}t} &= \sin(z) - by
#   \\
#   \frac{\mathrm{d}z}{\mathrm{d}t} &= \sin(x) - bz
# $$
#
# For $b<0.208186$, this system has chaotic solutions.
# At this value, the spectra of the solutions deviate from the Exponential Tail model.
# For smaller values, $0 < b < 0.2$,
# the Exponential Tail model is applicable:
# as $b$ decreases,
# the autocorrelation diverges and
# a larger part of the spectrum can be fitted.
# This is shown below by computing the error of the mean of numerical solutions.
#
# For $b=0$, the solutions become random walks with anomalous diffusion
# {cite:p}`rowlands_2008_simple`.
# In this case, it makes more sense to work with
# the spectrum of the time direvative of the solutions.
# However, due to the anomalous diffusion, the spectrum of these derivatives
# cannot be approximated well with the Exponential Tail model.
#
# This example is fully self-contained:
# input data is generated with numerical integration and then analyzed with Stacie.
# Dimensionless units are used throughout.

# %% [markdown]
# ## Import Libraries and Configure `matplotlib`

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from stacie import UnitConfig, compute_spectrum, estimate_acint
from stacie.plot import (
    plot_criterion,
    plot_fitted_spectrum,
    plot_spectrum,
)

# %%
mpl.rc_file("matplotlibrc")
# %config InlineBackend.figure_formats = ["svg"]

# %% [markdown]
# ## Data Generation
# The following cell implements the numerical integration of the oscillator
# for 100 different initial configurations.

# %%
NSYS = 100
NDIM = 3
NSTEP = 20000
TIMESTEP = 0.3


def time_derivatives(state: ArrayLike, *, b: float = 0.1) -> NDArray:
    """Compute the time derivatives defining the differential equations."""
    return np.sin(np.roll(state, 1)) - b * state


def integrate(state: ArrayLike, nstep: int, h: float) -> NDArray:
    """Integrate the System with Ralston's method, using a fixed time step h."""
    trajectory = np.zeros((nstep, *state.shape))
    for istep in range(nstep):
        k1 = time_derivatives(state)
        k2 = time_derivatives(state + (2 * h / 3) * k1)
        state += h * (k1 + 3 * k2) / 4
        trajectory[istep] = state
    return trajectory


def generate():
    """Generate solutions for random initial states."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, (NDIM, NSYS))
    return integrate(x, NSTEP, TIMESTEP)


trajectory = generate()


# %% [markdown]
#
# The solutions shown below are smooth, but for low enough values of $b$,
# they are pseudo-random on longer time scales.
# %%
def plot_traj(nplot=500):
    """Show the first 500 steps of the first 10 solutions."""
    fig, ax = plt.subplots()
    times = np.arange(nplot) * TIMESTEP
    ax.plot(times, trajectory[:nplot, 0, :10])
    ax.set_xlabel("Time [1]")
    ax.set_ylabel("$x(t)$")
    ax.set_title("Solutions")


plot_traj()

# %% [markdown]
# ## Spectrum
#
# In the chaotic regime, the low-frequency spectrum indicates diffusive motion:
# a large peak at the origin.
# The spectrum is normalized so that the autocorrelation integral
# becomes the variance of the mean.

# %%
uc = UnitConfig(acint_fmt=".2e", acint_unit_str="1", time_unit_str="1", freq_unit_str="1")
sequences = trajectory[:, 0, :].T
spectrum = compute_spectrum(
    sequences,
    timestep=TIMESTEP,
    prefactor=1 / (NSTEP * TIMESTEP * NSYS),
    include_zero_freq=False,
)
fig, ax = plt.subplots()
plot_spectrum(ax, uc, spectrum, nplot=500)

# %% [markdown]
# ## Error of the Mean
#
# The following cells fit the Exponential Tail model to the spectrum
# to derive the variance of the mean.

# %%
result = estimate_acint(spectrum, verbose=True)
fig, ax = plt.subplots()
plot_fitted_spectrum(ax, uc, result)
fig, ax = plt.subplots()
plot_criterion(ax, uc, result)

# %% [markdown]
# Due to the symmetry of the oscillator, the mean of the solutions should be zero.
# Within the uncertainty, this is indeed the case for the numerical solutions.

# %%
mean = sequences.mean()
print(f"Mean: {mean:.3e}")
error_mean = np.sqrt(result.acint)
print(f"Error of the mean: {error_mean:.3e}")

# %% [markdown]
# For sufficiently small values of $b$, the autocorrelation function is a simple
# exponentially decaying function, so that the two
# [autocorrelation times](../theory/properties/autocorrelation_time.md)
# are very similar:

# %%
print(f"corrtime_exp = {result.corrtime_exp:.3f} ± {result.corrtime_exp_std:.3f}")
print(f"corrtime_int = {result.corrtime_int:.3f} ± {result.corrtime_int_std:.3f}")


# %%  [markdown]
# ## Regression tests
#
# If you experiment with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(result.acint - 2.4772e-4) > 1e-7:
    raise ValueError(f"Wrong acint: {result.acint:.4e}")
if abs(result.corrtime_exp - 9.7475) > 1e-3:
    raise ValueError(f"Wrong corrtime_exp: {result.corrtime_exp:.4e}")
