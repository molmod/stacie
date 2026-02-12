# %% [markdown]
# # Quasi-Gaussian Quadrature (QGQ) Example
#
# This notebook illustrates how to use the module `stacie.qgq` to compute
# integrals over noisy integrands.
# While this is currently not the main goal of STACIE,
# it is often a useful step for postprocessing transport properties
# computed with STACIE at different densities and/or total energies.
#
# This QGQ method generates quadrature rules of the following form:
#
# $$
#   \int f(x) p(x) dx \approx \mathcal{I}_\text{CGC} = \frac{1}{N} \sum_{n=1}^N f(x_n)
# $$
#
# where $p(x)$ is a probability density function and the quadrature points
# are optimized to make the approximation exact for polynomials of a given degree.
# Unlike standard quadrature rules, the weights are all equal,
# meaning that the variance of the integral is minimal when the integrand
# has independent and identically distributed noise:
#
# $$
#   \var[\mathcal{I}_\text{CGC}] = \frac{1}{N} \var[f(x)]
# $$

# %% [markdown]
# ## Test Case With a Noisy Integrand
#
# This notebook demonstrates the QGQ method with a computationally cheap but noisy integrand.
# In practice, the integrand is typically (very) expensive to evaluate,
# and the noise arises from the sampling required to estimate the integrand at each point.
#
# We will use a noisy cosine function as our integrand,
# and the standard normal distribution as our probability density function:
#
# $$
#   \begin{aligned}
#     f(x) &= \cos(x) + \epsilon \\
#     p(x) &= \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right)
#   \end{aligned}
# $$
#
# The noise $\epsilon$ is drawn from a normal distribution with mean zero and standard deviation $\sigma$,
# and the calculations below use different values of $\sigma$ to illustrate the effect of noise.
#
# The true value of the integral (without noise) is given by $1/\sqrt{e}$.

# %% [markdown]
# ## Four Quadrature Methods
#
# We will compare four quadrature methods:
#
# 1. **Gauss-Hermite Quadrature**:
#    The standard Gauss-Hermite quadrature points and weights are optimal
#    for integrating polynomials times a normal distribution.
#    It is exact for polynomials of degree up to $2N-1$ when using $N$ points.
#    However, it does not account for noise in the integrand.
# 2. **QGQ with exact moments**:
#    We use the first 7 moments (0 to 6) of the standard normal distribution
#    to compute the quadrature points.
#    This method will give exact results for polynomials of degree up to 6,
#    and is expected to be more robust against noise.
# 3. **QGQ with approximate moments**:
#    We use the first 7 moments (0 to 6) of the standard normal distribution
#    to compute the quadrature points.
#    Instead of using the exact moments, we estimate them from 10000 samples
#    drawn from the standard normal distribution.
#    This mimics the situation where the moments are estimated from a finite number of samples.
# 4. **Monte Carlo**:
#    We draw grid points from the standard normal distribution and use them
#    to compute the integral using the standard Monte Carlo method.
#    This could be considered as a quadrature method with random points and equal weights.
#
# In all cases, we will use 30 quadrature points to compute the integral.
# The uncertainty propagation of $\epsilon$ can be performed analytically:
#
# $$
#   \sigma_\mathcal{I} = \sigma \sqrt{\sum_{n=1}^N w_i^2}
# $$
#
# where $w_i$ are the quadrature weights.
#

# %% [markdown]
# ## Implementation
#

# %%
import numpy as np
import matplotlib.pyplot as plt
from stacie.qgq import construct_qgq_stdnormal, construct_qgq_empirical

# %%
# Quadrature grids and weights
x_gh, w_gh = np.polynomial.hermite_e.hermegauss(30)
w_gh /= np.sqrt(2 * np.pi)
x_qgq_exact, w_qgq_exact = construct_qgq_stdnormal(np.linspace(0.1, 1.0, 15), 6)
rng = np.random.default_rng(seed=0)
std_points = rng.standard_normal(size=10000)
x_qgq_approx, w_qgq_approx = construct_qgq_empirical(std_points, 30, 6)
x_mc = rng.standard_normal(size=30)
x_mc.sort()


# %%
# Plot of the quadrature points and weights
def plot_points_weights():
    plt.close("quad")
    _, ax = plt.subplots(num="quad")
    ax.plot(x_gh, w_gh, "+", label="Gauss-Hermite")
    ax.plot(x_qgq_exact, w_qgq_exact, "+", label="QGQ Exact")
    ax.plot(x_qgq_approx, w_qgq_approx, "+", label="QGQ Approx")
    ax.plot(x_mc, np.full_like(x_mc, 1 / len(x_mc)), "+", label="Monte Carlo")
    ax.set_xlabel("Quadrature points")
    ax.set_ylabel("Quadrature weights")
    ax.legend()
    ax.set_title("Quadrature points and weights")


plot_points_weights()


# %%
# Plot of the quadrature points and weights
def plot_points():
    plt.close("points")
    _, ax = plt.subplots(num="points")
    idx = np.arange(len(x_gh))
    ax.plot(x_gh, idx, "+", label="Gauss-Hermite")
    ax.plot(x_qgq_exact, idx, "+", label="QGQ Exact")
    ax.plot(x_qgq_approx, idx, "+", label="QGQ Approx")
    ax.plot(x_mc, idx, "+", label="Monte Carlo")
    ax.set_xlabel("Quadrature points")
    ax.set_ylabel("Index")
    ax.legend()
    ax.set_title("Quadrature points and weights")


plot_points()


# %%
# Numerical integration
integral_true = 1 / np.sqrt(np.e)
integral_gh = np.dot(w_gh, np.cos(x_gh))
integral_qgq_exact = np.dot(w_qgq_exact, np.cos(x_qgq_exact))
integral_qgq_approx = np.dot(w_qgq_approx, np.cos(x_qgq_approx))
integral_mc = np.cos(x_mc).mean()

print(f"True value:    {integral_true:.15f}")
print(f"Gauss-Hermite: {integral_gh:.15f}")
print(f"QGQ Exact:     {integral_qgq_exact:.15f}")
print(f"QGQ Approx:    {integral_qgq_approx:.15f}")
print(f"Monte Carlo:   {integral_mc:.15f}")

# %% [markdown]
#
# Note that the deviations between the first two quadrature methods and the true value
# are not due to noise, but rather due to the fact that
# the quadrature rules are approximate for a non-polynomial integrand.
#
# The approximate QGQ method and Monte Carlo are affected by sampling noise:
# - In case of the approximate QGQ method,
#   the moments are estimated from a finite number of samples,
#   which introduces noise in the quadrature points.
# - In case of Monte Carlo, the quadrature points are drawn randomly,
#   which also introduces noise in the integral estimate.


# %%
# Plot the effect of integrand noise on the integral estimates


def plot_sigma():
    sigmas = np.linspace(0, 0.05, 10)
    plt.close("sigma")
    fig, ax = plt.subplots(num="sigma")
    ax.axhline(integral_true, color="k", lw=5, alpha=0.8, label="True value")

    # Gauss-Hermite
    ax.axhline(integral_gh, color="C0", label="Gauss-Hermite")
    factor_gh = np.sqrt((w_gh**2).sum())
    ax.fill_between(
        sigmas,
        integral_gh - factor_gh * sigmas,
        integral_gh + factor_gh * sigmas,
        color="C0",
        alpha=0.3,
    )

    # QGQ Exact
    ax.axhline(integral_qgq_exact, color="C1", label="QGQ Exact")
    factor_qgq_exact = np.sqrt((w_qgq_exact**2).sum())
    ax.fill_between(
        sigmas,
        integral_qgq_exact - factor_qgq_exact * sigmas,
        integral_qgq_exact + factor_qgq_exact * sigmas,
        color="C1",
        alpha=0.3,
    )

    # QGQ Approx
    ax.axhline(integral_qgq_approx, color="C2", label="QGQ Approx")
    factor_qgq_approx = np.sqrt((w_qgq_approx**2).sum())
    ax.fill_between(
        sigmas,
        integral_qgq_approx - factor_qgq_approx * sigmas,
        integral_qgq_approx + factor_qgq_approx * sigmas,
        color="C2",
        alpha=0.3,
    )

    # Plot details
    ax.set_xlabel(r"Standard deviation $\sigma$ of the noise on the integrand")
    ax.set_ylabel("Integral estimate")
    ax.legend()
    ax.set_title("Effect of integrand noise on integral estimates")


plot_sigma()

# %% [markdown]
#
# The plot shows that even for errors on the integrand on the order of a percentage,
# it is beneficial to use the QGQ method instead of standard quadrature rules or Monte Carlo,
# as it significantly reduces the uncertainty on the integral estimate.
