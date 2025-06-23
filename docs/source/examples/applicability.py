# %% [markdown]
# # Applicability of the Lorentz Model
#
# STACIE's [Lorentz model](#section-lorentz-target) assumes that
# the autocorrelation function decays exponentially for large lag times.
# Not all dynamical systems exhibit this exponential relaxation.
# If you want to apply STACIE to systems without exponential relaxation,
# you can use the [exppoly model](#section-exppoly-target) instead.
#
# To illustrate the applicability of the Lorentz model,
# this notebook applies STACIE to numerical solutions of
# [Thomas' Cyclically Symmetric Attractor](https://en.wikipedia.org/wiki/Thomas%27_cyclically_symmetric_attractor):
#
# $$
#   \begin{aligned}
#     \frac{\mathrm{d}x}{\mathrm{d}t} &= \sin(y) - bx
#     \\
#     \frac{\mathrm{d}y}{\mathrm{d}t} &= \sin(z) - by
#     \\
#     \frac{\mathrm{d}z}{\mathrm{d}t} &= \sin(x) - bz
#   \end{aligned}
# $$
#
# For $b<0.208186$, this system has chaotic solutions.
# As a result, the system looses memory of its initial conditions rather quickly,
# and the autocorrelation function tends to decay exponentially.
# At the boundary, $b=0.208186$, the exponential decay is no longer valid
# and the spectrum deviates from the Lorentzian shape.
# In practice, the Lorentz model is applicable for smaller values, $0 < b < 0.17$.
#
# For $b=0$, the solutions become random walks with anomalous diffusion
# {cite:p}`rowlands_2008_simple`.
# In this case, it makes more sense to work with
# the spectrum of the time derivative of the solutions.
# However, due to the anomalous diffusion, the spectrum of these derivatives
# cannot be approximated well with the Lorentz model.
#
# This example is fully self-contained:
# input data is generated with numerical integration and then analyzed with STACIE.
# Dimensionless units are used throughout.
#
# We suggest you experiment with this notebook by changing the $b$ parameter
# and replacing the Lorentz model with the ExpPoly model.


# %% [markdown]
# ## Library Imports and Matplotlib Configuration

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from stacie import (
    UnitConfig,
    compute_spectrum,
    estimate_acint,
    LorentzModel,
    plot_extras,
    plot_fitted_spectrum,
    plot_spectrum,
)

# %%
mpl.rc_file("matplotlibrc")
# %config InlineBackend.figure_formats = ["svg"]

# %% [markdown]
# ## Data Generation
# The following cell implements the numerical integration of the oscillator
# using [Ralston's method](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Ralston's_method)
# for 100 different initial configurations.
# The parameter $b$ is given as an argument to the `generate()` function
# at the last line of the next cell.

# %%
NSYS = 100
NDIM = 3
NSTEP = 20000
TIMESTEP = 0.3


def time_derivatives(state: ArrayLike, b: float) -> NDArray:
    """Compute the time derivatives defining the differential equations."""
    return np.sin(np.roll(state, 1, axis=1)) - b * state


def integrate(state: ArrayLike, nstep: int, h: float, b: float) -> NDArray:
    """Integrate the System with Ralston's method, using a fixed time step h.

    Parameters
    ----------
    state
        The initial state of the system, shape `(ndim, nsys)`,
        where `ndim` is the number of dimensions and `nsys` systems to integrate in parallel.
    nstep
        The number of time steps to integrate.
    h
        The time step size.
    b
        The parameter $b$ in the differential equations.

    Returns
    -------
    trajectory
        The trajectory of the system, shape `(nstep, ndim, nsys)`.
        The first dimension is the time step, the second dimension is the state variable,
        and the third dimension is the system index.
    """
    trajectory = np.zeros((nstep, *state.shape))
    for istep in range(nstep):
        k1 = time_derivatives(state, b)
        k2 = time_derivatives(state + (2 * h / 3) * k1, b)
        state += h * (k1 + 3 * k2) / 4
        trajectory[istep] = state
    return trajectory


def generate(b: float):
    """Generate solutions for random initial states."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, (NDIM, NSYS))
    return integrate(x, NSTEP, TIMESTEP, b)


trajectory = generate(b=0.1)


# %% [markdown]
#
# The solutions shown below are smooth, but for low enough values of $b$,
# they are pseudo-random over longer time scales.
# %%
def plot_traj(nplot=500):
    """Show the first 500 steps of the first 10 solutions."""
    plt.close("traj")
    _, ax = plt.subplots(num="traj")
    times = np.arange(nplot) * TIMESTEP
    ax.plot(times, trajectory[:nplot, 0, 0], label="$x(t)$")
    ax.plot(times, trajectory[:nplot, 1, 0], label="$y(t)$")
    ax.plot(times, trajectory[:nplot, 2, 0], label="$z(t)$")
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")
    ax.set_title(f"Example solutions (first {nplot} steps)")
    ax.legend()


plot_traj()

# %% [markdown]
# ## Spectrum
#
# In the chaotic regime, the low-frequency spectrum indicates diffusive motion:
# a large peak at the origin.
# The spectrum is normalized so that the autocorrelation integral
# becomes the [variance of the mean](../properties/error_estimates.md).

# %%
uc = UnitConfig(acint_fmt=".2e")
sequences = trajectory[:, 0, :].T  # use x(t) only
spectrum = compute_spectrum(
    sequences,
    timestep=TIMESTEP,
    prefactors=2.0 / (NSTEP * TIMESTEP * NSYS),
    include_zero_freq=False,
)
plt.close("spectrum")
_, ax = plt.subplots(num="spectrum")
plot_spectrum(ax, uc, spectrum, nplot=500)

# %% [markdown]
# Note that we only use component 0, i.e. $x(t)$, of each system as input for the spectra.
# This ensures that fully independent sequences are used in the analysis below,
# which is assumed by the statistical model of the spectrum used by STACIE.

# %% [markdown]
# ## Error of the Mean
#
# The following cells fit the Lorentz model to the spectrum
# to derive the variance of the mean.

# %%
result = estimate_acint(spectrum, LorentzModel(), verbose=True)

# %% [markdown]
# Due to the symmetry of the oscillator, the mean of the solutions should be zero.
# Within the {term}`uncertainty`, this is indeed the case for the numerical solutions,
# as shown below.

# %%
mean = sequences.mean()
print(f"Mean: {mean:.3e}")
error_mean = np.sqrt(result.acint)
print(f"Error of the mean: {error_mean:.3e}")

# %% [markdown]
# For sufficiently small values of $b$, the autocorrelation function
# decays exponentially, so that the two
# [autocorrelation times](../properties/autocorrelation_time.md)
# are very similar:

# %%
print(f"corrtime_exp = {result.corrtime_exp:.3f} ± {result.corrtime_exp_std:.3f}")
print(f"corrtime_int = {result.corrtime_int:.3f} ± {result.corrtime_int_std:.3f}")

# %% [markdown]
# To further gauge the applicability of the Lorentz model,
# it is useful to plot the fitted spectrum and the intermediate results
# as a function of the cutoff frequency, as shown below.

# %%
plt.close("fitted")
fig, ax = plt.subplots(num="fitted")
plot_fitted_spectrum(ax, uc, result)
plt.close("extras")
fig, axs = plt.subplots(2, 2, num="extras")
plot_extras(axs, uc, result)

# %% [markdown]
# It is clear that at higher cutoff frequencies, which are given a negligible weight,
# the spectrum deviates from the Lorentzian shape.
# Hence, at shorter time scales, the autocorrelation function does not decay exponentially.
# This was to be expected, as the input sequences are smooth functions.
# To further confirm this, we recommend rerunning this notebook with different values of $b$:
#
# - For lower value, such as $b=0.05$, the Lorentz model will fit the spectrum better,
#   which is reflected in lower Z-score values.
# - Up to $b=0.17$, the Lorentz model is still applicable, but the Z-scores will increase.
# - For $b=0.2$, the Lorentz model will not be able to assign an exponential correlation time.
#   To be able to run the notebook until the last plot, you need to comment out the line
#   that prints the exponential correlation time.

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(result.acint - 2.47e-4) > 2e-5:
    raise ValueError(f"Wrong acint: {result.acint:.4e}")
if abs(result.corrtime_exp - 10.408) > 1e-1:
    raise ValueError(f"Wrong corrtime_exp: {result.corrtime_exp:.4e}")
