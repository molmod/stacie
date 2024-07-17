#!/usr/bin/env python

# %% [markdown]
# # Risk Metric Demo
#
# This notebook illustrates the risk metric for detecting underfitting,
# using a simple polynomial fit to a cosine function as test case.
# To get an intuition of how the method works,
# we suggest you try making the following changes:
#
# - Increase the polynomial degree.
# - Increase the noise (sigma).
# - Override `icut_best` to visualize over- and underfitted results.
# - Change the random seed.

# %% [markdown]
# ## Import Libraries and Configure `matplotlib`

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from stacie.riskmetric import risk_metric_cumsum

# %%
mpl.rc_file("matplotlibrc")
# %config InlineBackend.figure_formats = ["svg"]

# %% [markdown]
# ## Example Data
#
# The data for the fitting problem is a cosine function plus some normally distributed noise.

# %%
rng = np.random.default_rng(42)
xgrid = np.linspace(0, 4 * np.pi, 100)
sigma = 0.1
ytruth = np.cos(xgrid)
yvalues = ytruth + rng.normal(0, sigma, len(xgrid))

# %% [markdown]
# ## Polynomial Fits
#
# A polynomial is fitted to a series of subsets of the data of increasing size.
# For each data cutoff, the risk metric is computed and
# the lowest risk is used in the following cells.


# %%
# Fit a polynomial to the data, using an increasing number of points.
def fit_poly(ncut, degree):
    """Fit a polynomial to the data up to ncut, using a Chebyshev basis."""
    dm = np.polynomial.chebyshev.chebvander(xgrid[:ncut], degree)
    pars = np.linalg.lstsq(dm, yvalues[:ncut])[0]
    return np.dot(dm, pars), pars


degree = 3
ncuts = np.arange(degree + 2, len(xgrid))
risks = np.zeros(ncuts.shape)
for icut, ncut in enumerate(ncuts):
    residuals = fit_poly(ncut, degree)[0] - yvalues[:ncut]
    risks[icut] = risk_metric_cumsum(residuals / sigma)
icut_best = risks.argmin()
ncut_best = ncuts[icut_best]
print("Optimal data cutoff =", ncut_best)

# %% [markdown]
# ## Risk Minimization

# %%
# Plot the risk as function of the cutoff.
fig, ax = plt.subplots()
ax.plot(ncuts, risks)
ax.axvline(ncut_best, color="k", alpha=0.3)
risk_scale = abs(risks.min())
ax.set_ylim(risks.min() - 0.1 * risk_scale, 2 * risk_scale)
ax.set_xlabel("Cutoff index")
ax.set_ylabel("Over- and underfitting risk metric")
ax.set_title("Risk Minimization")

# %% [markdown]
# ## Results

# %%
# Plot the fit over the whole domain.
pars = fit_poly(ncut_best, degree)[1]
ymodel = np.dot(np.polynomial.chebyshev.chebvander(xgrid, degree), pars)
fig, ax = plt.subplots()
ax.plot(xgrid, ytruth, label="truth")
ax.plot(xgrid, yvalues, "k.", label="data")
ax.plot(xgrid, ymodel, label="polynomial fit")
ax.axvline(xgrid[ncut_best], color="k", alpha=0.3)
ax.set_ylim(-1 - 2 * sigma, 1 + 2 * sigma)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Data and Models")
ax.legend(loc=0)

# %%
# Plot the residuals over the whole domain.
# Note that the first discarded point is the one at the vertical cutoff line.
fig, ax = plt.subplots()
ax.plot(xgrid, yvalues - ytruth, "C0.", label="true residuals")
ax.plot(xgrid, yvalues - ymodel, "C1.", label="fit residuals")
ax.axvline(xgrid[ncut_best], color="k", alpha=0.3)
ax.axhline(-sigma, color="k", alpha=0.3)
ax.axhline(sigma, color="k", alpha=0.3)
ax.set_ylim(-10 * sigma, 10 * sigma)
ax.set_xlabel("x")
ax.set_ylabel("Δy")
ax.set_title("Residuals")
ax.legend(loc=0)

# %%
# Plot individual cumulative sums used in the risk metric.
# Note that the first discarded point is the one at the vertical cutoff line.
fig, ax = plt.subplots()
cs = np.zeros(ncut_best + 1)
nr = (yvalues - ymodel)[:ncut_best] / sigma
cs = []
for j in range(ncut_best + 1):
    cs.append(nr[:j].sum() - nr[j:].sum())
cs = np.array(cs)
ax.plot(cs**2, "C0.")
ax.axhline(ncut_best - 4, color="k", alpha=0.3)
ax.set_xlabel("index $i$")
ax.set_ylabel(r"$\hat{U}^2_i$")
ax.set_title("Shifted Cumulative Sums")

# %% [markdown]
# Note that the cumulative sums fall well below the expectation value for standard normal residuals.
# There are several reasons for this:
# - The variance of the regression residuals is $\frac{N_\text{res} - \nu}{N_\text{res}}\sigma$,
#   where $\nu$ is the number of fitted parameters.
#   The risk metric ignores $\nu$.
#   (Including it would only make a small difference.)
# - Regression residuals are anticorrelated, meaning that the sum of the residuals
#   has a lower variance than that of independent random variables with the same variance.
