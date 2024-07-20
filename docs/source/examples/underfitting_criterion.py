#!/usr/bin/env python

# %% [markdown]
# # Underfitting Criterion Demo
#
# This notebook illustrates the criterion for detecting underfitting,
# using a simple polynomial fit to a cosine function as test case.
# To get an intuition of how the method works,
# we suggest you try making the following changes:
#
# - Increase the polynomial degree.
# - Increase the noise (sigma).
# - Override `ifit_best` to visualize over- and underfitted results.
# - Change the random seed.

# %% [markdown]
# ## Import Libraries and Configure `matplotlib`

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from stacie.cutoff import general_ufc

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
# For each data cutoff, the underfitting criterion is computed and
# results for the lowest criterion are used in the following cells.


# %%
# Fit a polynomial to the data, using an increasing number of points.
def fit_poly(nfit, degree):
    """Fit a polynomial to the data up to nfit, using a Chebyshev basis."""
    dm = np.polynomial.chebyshev.chebvander(xgrid[:nfit], degree)
    pars = np.linalg.lstsq(dm, yvalues[:nfit])[0]
    return np.dot(dm, pars), pars


degree = 3
npar = degree + 1
nfits = np.arange(degree + 2, len(xgrid))
ufcs = np.zeros(nfits.shape)
nrvars = np.zeros(nfits.shape)
aics = np.zeros(nfits.shape)
for ifit, nfit in enumerate(nfits):
    normalized_residuals = (fit_poly(nfit, degree)[0] - yvalues[:nfit]) / sigma
    ufcs[ifit] = general_ufc(normalized_residuals)
    nrvars[ifit] = (normalized_residuals**2).mean()
    aics[ifit] = (normalized_residuals**2).sum() + 2 * (len(xgrid) - nfit)
ifit_best = ufcs.argmin()  # <--- change to override the cutoff
nfit_best = nfits[ifit_best]
print("ifit_best =", ifit_best)
print("nfit_best =", nfit_best)

# %% [markdown]
# ## Underfitting Criterion Minimization


# %%
# Plot the underfitting criterion as function of the cutoff.
def plot_underfitting():
    nplot = 42
    fig, ax1 = plt.subplots()
    ax1.plot(nfits[:nplot], ufcs[:nplot] - ufcs.min(), color="C2", label="UFC")
    ax1.plot(nfits[:nplot], aics[:nplot] - aics.min(), color="C0", label="AIC")
    ax1.axvline(nfit_best, color="k", alpha=0.3)
    ax1.set_ylim(0)
    ax1.set_xlabel("Sample size (nfit)")
    ax1.set_ylabel("(Shifted) Criterion [1]")
    ax1.tick_params(axis="y")
    ax1.legend(loc=0)
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(nfits[:nplot], nrvars[:nplot], color="C3")
    ax2.plot(nfits[:nplot], np.sqrt((nfits - npar) / nfits)[:nplot], color="C3", ls="--")
    ax2.set_ylim(0, 2)
    ax2.set_ylabel("Sampling variance normalized residuals [1]", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")


plot_underfitting()

# %% [markdown]
# The red dashed line show the expectation value of the variance of the normalized residuals
# in case of linear regression.
# As this example illustrates, there is no simple way of selecting the amount of data by directly
# analyzing the variance of the residuals:
#
# - There are multiple sample sizes for which the sampling variance of the residuals is close
#   to the expected value. Which one should be used?
#   Try setting `ifit_best = 32` above, as a reasonable choice and rerun the notebook.
#   In this case, the plot of the residuals below exhibits an oscillatory trend.
#
# - When the sampling variance clearly exceeds the expected value, e.g. becomes 2,
#   the model is also clearly overfitting.
#   For example, try `ifit_best = 38`.
#
# In comparison, the minimizer of the underfitting criterion (`ifit_best = 27`) is fairly robust.
# For this cutoff, the residuals show no apparent trend.
# For this example, also the minimizer of the AIC (`ifit_best = 26`) would be a good cutoff value.

# %%

# %% [markdown]
# ## Results


# %%
# Plot the fit over the whole domain.
pars = fit_poly(nfit_best, degree)[1]
ymodel = np.dot(np.polynomial.chebyshev.chebvander(xgrid, degree), pars)


def plot_model_and_data():
    fig, ax = plt.subplots()
    ax.plot(xgrid, ytruth, label="truth")
    ax.plot(xgrid, yvalues, "k.", label="data")
    ax.plot(xgrid, ymodel, label="polynomial fit")
    ax.axvline(xgrid[nfit_best], color="k", alpha=0.3)
    ax.set_ylim(-1 - 2 * sigma, 1 + 2 * sigma)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Data and Models")
    ax.legend(loc=0)


plot_model_and_data()


# %%
# Plot the residuals over the whole domain.
# Note that the first discarded point is the one at the vertical cutoff line.
def plot_residuals():
    fig, ax = plt.subplots()
    ax.plot(xgrid, yvalues - ytruth, "C0.", label="true residuals")
    ax.plot(xgrid, yvalues - ymodel, "C1.", label="fit residuals")
    ax.axvline(xgrid[nfit_best], color="k", alpha=0.3)
    ax.axhline(-sigma, color="k", alpha=0.3)
    ax.axhline(sigma, color="k", alpha=0.3)
    ax.set_ylim(-10 * sigma, 10 * sigma)
    ax.set_xlabel("x")
    ax.set_ylabel("Δy")
    ax.set_title("Residuals")
    ax.legend(loc=0)


plot_residuals()


# %%
# Plot individual cumulative sums used in the underfitting criterion.
# Note that the first discarded point is the one at the vertical cutoff line.
def plot_cumulative_sums():
    fig, ax = plt.subplots()
    cs = np.zeros(nfit_best + 1)
    nr = (yvalues - ymodel)[:nfit_best] / sigma
    cs = []
    for j in range(nfit_best + 1):
        cs.append(nr[:j].sum() - nr[j:].sum())
    cs = np.array(cs)
    ax.plot(cs**2, "C0.")
    ax.axhline(nfit_best - (degree + 1), color="k", alpha=0.3)
    ax.set_xlabel("index $i$")
    ax.set_ylabel(r"$\hat{U}^2_i$")
    ax.set_title("Shifted Cumulative Sums")


plot_cumulative_sums()

# %% [markdown]
# The squared cumulative sums are well below the upper limit of the expectation value.
# Regression residuals are anticorrelated, meaning that summed residuals
# have a lower variance than a sum of independent random variables with the same variance.

# %%
# Basic regression tests to verify that the results in the notebook do not change unexpectedly.

if nfit_best != 32:
    raise ValueError(f"Wrong nfit_best: {nfit_best}")
