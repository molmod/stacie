# %% [markdown]
# # Applicability of the Pade Model
#
# STACIE's [Pade model](#section-pade-target) with $S_\text{num}=\{0, 2\}$ and $S_\text{den}=\{2\}$ assumes that
# the autocorrelation function decays exponentially for large lag times.
# Not all dynamical systems exhibit this exponential relaxation.
# If you want to apply STACIE to systems without exponential relaxation,
# you can use the [exppoly model](#section-exppoly-target) instead.
#
# To illustrate the applicability of the Pade model,
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
# At the boundary, $b=0.208186$, the exponential decay is no longer valid and the spectrum deviates from the Lorentzian shape.
# In practice, the Pade model is applicable for smaller values, $0 < b < 0.2$.
#
# For $b=0$, the solutions become random walks with anomalous diffusion
# {cite:p}`rowlands_2008_simple`.
# In this case, it makes more sense to work with
# the spectrum of the time derivative of the solutions.
# However, due to the anomalous diffusion, the spectrum of these derivatives
# cannot be approximated well with the Pade model.
#
# This example is fully self-contained:
# input data is generated with numerical integration and then analyzed with STACIE.
# Dimensionless units are used throughout.
#
# We suggest you experiment with this notebook by changing the $b$ parameter
# and replacing the Pade model with the ExpPoly model.


# %% [markdown]
# ## Library Imports and Matplotlib Configuration

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from stacie import UnitConfig, compute_spectrum, estimate_acint, PadeModel
from stacie.plot import plot_extras, plot_fitted_spectrum, plot_spectrum

# %%
mpl.rc_file("matplotlibrc")
# %config InlineBackend.figure_formats = ["svg"]

# %% [markdown]
# ## Data Generation
# The following cell implements the numerical integration of the oscillator
# using [Ralston's method](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Ralston's_method)
# for 100 different initial configurations.

# %%
NSYS = 100
NDIM = 3
NSTEP = 20000
TIMESTEP = 0.3


def time_derivatives(state: ArrayLike, b: float) -> NDArray:
    """Compute the time derivatives defining the differential equations."""
    return np.sin(np.roll(state, 1, axis=1)) - b * state


def integrate(state: ArrayLike, nstep: int, h: float, b: float) -> NDArray:
    """Integrate the System with Ralston's method, using a fixed time step h."""
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
# becomes the [variance of the mean](../properties/error_estimates.md).

# %%
uc = UnitConfig(acint_fmt=".2e", acint_unit_str="1", time_unit_str="1", freq_unit_str="1")
sequences = trajectory[:, 0, :].T
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
# ## Error of the Mean
#
# The following cells fit the Pade model to the spectrum
# to derive the variance of the mean.

# %%
result = estimate_acint(spectrum, PadeModel([0, 2], [2]), verbose=True)
plt.close("fitted")
fig, ax = plt.subplots(num="fitted")
plot_fitted_spectrum(ax, uc, result)
plt.close("extras")
fig, axs = plt.subplots(2, 2, num="extras")
plot_extras(axs, uc, result)

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
# [autocorrelation times](../properties/autocorrelation_time.md)
# are very similar:

# %%
print(f"corrtime_exp = {result.corrtime_exp:.3f} ± {result.corrtime_exp_std:.3f}")
print(f"corrtime_int = {result.corrtime_int:.3f} ± {result.corrtime_int_std:.3f}")

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(result.acint - 2.47e-4) > 2e-5:
    raise ValueError(f"Wrong acint: {result.acint:.4e}")
if abs(result.corrtime_exp - 10.018) > 1e-1:
    raise ValueError(f"Wrong corrtime_exp: {result.corrtime_exp:.4e}")
