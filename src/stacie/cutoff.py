# Stacie is a STable AutoCorrelation Integral Estimator.
# Copyright (C) 2024-2025 The contributors of the Stacie Python Package.
# See the CONTRIBUTORS.md file in the project root for a full list of contributors.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# --
"""Criteria for selecting the part of the spectrum to fit to.

All ``*_criterion`` functions in this module have the same API:
they receive a dictionary of properties computed in ``estimate.estimate_acint``
and return a dictionary with at least one key ``"criterion"``.
The lower the value of this criterion,
the better the cutoff balances between over- and underfitting.

Some function also add a field ``"criterion_expected"`` to the result,
with the expected value of the criterion.

The functions may also include a field ``"criterion_scale"``.
This is a suitable order of magnitude for the range of the y-axis, which is used for plotting
"""

from collections.abc import Callable
from typing import NewType

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, gammaln

__all__ = (
    "CutoffCriterion",
    "akaike_criterion",
    "entropy_criterion",
    "expected_ufc",
    "general_ufc",
    "halfapprox_criterion",
    "halfhalf_criterion",
    "sumsq_criterion",
    "underfitting_criterion",
)


CutoffCriterion = NewType("CutoffCriterion", Callable[[dict[str, NDArray]], float])


def mark_criterion(*, half_opt: bool = False):
    """Add metadata to a cutoff criterion function.

    Parameters
    ----------
    half_opt
        Whether the criterion requires an optimization of parameters
        for the first or second half of the spectrum.
    """

    def decorator(criterion: CutoffCriterion) -> CutoffCriterion:
        criterion.half_opt = half_opt
        return criterion

    return decorator


@mark_criterion()
def entropy_criterion(props: dict[str, np.ndarray]) -> float:
    r"""
    Compute the entropy criterion based on the negative log Wiener entropy (NLWE).
    In this case, the NLWE is computed using the following formula:

    .. math::

        \text{NLWE} = -\ln(\text{WE}) = \ln(\text{AM}) - \ln(\text{GM})

    where AM and GM are the arithmetic and geometric mean, respectively,
    of the spectrum amplitudes divided by the model of the spectrum.

    The expected value of the NLWE can be derived using the properties of the Gamma distribution:

    .. math::

        \mathrm{E}[\text{NLWE}] = \psi(n \kappa) - \phi(\kappa) - \ln(n)

    where :math:`\psi` is the digamma function, :math:`n` is the number of frequencies
    and :math:`\kappa` is the shape parameter of the Gamma distribution.

    The criterion is then computed as:

    .. math::

        \text{criterion} = \text{NLWE}_{\text{empirical}} - \text{NLWE}_{\text{expected}} \ln(n)

    Without the second term, the NLWE tends to be low and noisy over a range of spectra,
    until it suddenly increase when the spectrum is underfitted.
    Including the second term, which is a slowly decreasing function of frequency, ensures that
    the criterion minimum shifts toward the highest possible frequency at which the
    spectrum is at most slightly underfitted.

    Parameters
    ----------
    props
        The property dictionary returned by the :py:meth:`stacie.cost.LowFreqCost.props` method.

    Returns
    -------
    results
        A dictionary with "criterion" and other fields.
        (See module docstring for details.)
    """
    ratio = props["amplitudes"] / props["amplitudes_model"][0]
    nfreq = len(props["freqs"])
    kappa = props["kappas"]

    nlwe_empirical = np.log(ratio.mean()) - np.log(ratio).mean()
    nlwe_expected = (digamma(nfreq * kappa) - digamma(kappa)).mean() - np.log(nfreq)

    return {
        "criterion": nlwe_empirical - nlwe_expected * np.log(nfreq),
        "criterion_expected": nlwe_expected * (1 - np.log(nfreq)),
        "criterion_scale": 2 * nlwe_expected,
    }


@mark_criterion()
def underfitting_criterion(props: dict[str, NDArray]) -> float:
    """Quantify the degree of underfitting of a smooth spectrum model to noisy data.

    See documentation for more details.

    Parameters
    ----------
    props
        The property dictionary returned by the :py:meth:`stacie.cost.LowFreqCost.props` method.

    Returns
    -------
    results
        A dictionary with "criterion" and other fields.
        (See module docstring for more details.)
    """
    residuals = (props["amplitudes"] / props["thetas"] - props["kappas"]) / np.sqrt(props["kappas"])
    return {
        "criterion": general_ufc(residuals),
        "criterion_expected": expected_ufc(props["amplitudes_model"][1]),
        "criterion_scale": len(props["amplitudes"]),
    }


def general_ufc(residuals: NDArray[float]) -> float:
    """A general-purpose implementation of the underfitting criterion.

    Parameters
    ----------
    residuals
        Normalized residuals,
        i.e. with the maximum likelihood estimate of the mean and standard deviation
        of the prediction at each point.

    Returns
    -------
    criterion
        A value quantifying the degree of underfitting.
        This criterion can be used to compare different selections of (contiguous) fitting data
        to which the same model is fitted.
        A lower value indicates better fit.
    """
    if residuals.ndim != 1:
        raise TypeError("The residuals must be a 1D array.")
    nfit = len(residuals)
    if nfit < 2:
        raise TypeError("The underfitting criterion requires at least two residuals.")
    # 'scs' is the abbreviation of 'symmetric cumulative sum'.
    scs = np.zeros(nfit + 1)
    np.cumsum(residuals, out=scs[1:])
    scs -= scs[-1] / 2
    scs *= 2
    return (scs**2).mean() - nfit


def expected_ufc(basis: NDArray[float]) -> float:
    """Compute the expected value of the underfitting criterion.

    Parameters
    ----------
    basis
        A set of basis functions, obtained by linearizing the regression problem.
        The array should have a shape of `(nparameters, nfreq)`, where each row
        represents a basis vector. The residuals must be orthogonal to these basis
        functions. Note that the basis vectors do not need to be orthogonal or normalized.

    Returns
    -------
    expected
        The expected value of the underfitting criterion
        (averaged over all possible residuals orthogonal to the basis functions).
    """
    nbasis, nfreq = basis.shape
    overlap = np.dot(basis, basis.T)
    evals, evecs = np.linalg.eigh(overlap)
    basis = np.dot(evecs.T, basis) / np.sqrt(evals.reshape(-1, 1))
    overlap = np.dot(basis, basis.T)

    # 'scs' is the abbreviation of 'symmetric cumulative sum'.
    scs_basis = np.zeros((nbasis, nfreq + 1))
    np.cumsum(basis, out=scs_basis[:, 1:], axis=1)
    scs_basis -= scs_basis[:, -1:] / 2
    scs_basis *= 2
    uvec_sqmodel = nfreq - np.sum(scs_basis**2, axis=0)
    return uvec_sqmodel.sum() / (nfreq + 1) - nfreq


@mark_criterion()
def sumsq_criterion(props: dict[str, np.ndarray]) -> float:
    """A cutoff criterion based on the statistics of the sum of squares of the residuals.

    The sum of (normalized) squared residuals in a regression problem is known
    to follow a chi-squared distribution with a number of degrees of freedom
    equal to the number of data points minus the number of parameters.

    This criterion consists of three terms:

      1. Minus the log-likelihood of the sum of squares of the residuals
         under the assumption that it follows this chi-squared distribution.
      2. The entropy of the same chi-squared distribution,
         which corresponds to the expected value of the first term.
      3. The log of the number of degrees of freedom.

    The sum of the first two terms is flat (with some statistical noise)
    until the spectrum is clearly underfitted, at which point it jumps up.
    The third term is added to shift the minimum to the highest frequency
    where the spectrum is at worst a little underfitted.

    Parameters
    ----------
    props
        The property dictionary returned by the :py:meth:`stacie.cost.LowFreqCost.props` method.

    Returns
    -------
    results
        A dictionary with "criterion" and other fields.
        (See module docstring for details.)
    """
    amplitudes = props["amplitudes"]
    kappas = props["kappas"]
    thetas = props["thetas"]
    residuals = (amplitudes / thetas - kappas) / np.sqrt(kappas)
    sumsq = (residuals**2).sum()
    ndof = len(residuals) - len(props["pars"])
    nlogprob = (
        0.5 * ndof * np.log(2)
        + gammaln(0.5 * ndof)
        + (1 - 0.5 * ndof) * np.log(sumsq)
        + 0.5 * sumsq
    )
    entropy = 0.5 * ndof + np.log(2) + gammaln(0.5 * ndof) + (1 - 0.5 * ndof) * digamma(0.5 * ndof)
    return {
        "criterion": nlogprob - entropy - np.log(ndof),
        "criterion_expected": -np.log(ndof),
        "criterion_scale": 2.0,
    }


@mark_criterion()
def akaike_criterion(props: dict[str, NDArray]) -> float:
    """Compute the Akaike Information Criterion (AIC) for the whole spectrum: fitted + discarded.

    The model used for the AIC is the one used for the spectrum fitting
    plus one additional free parameter per frequency past the cutoff.
    The log-likelihood and the number of parameters are computed for the whole spectrum.

    Parameters
    ----------
    props
        The property dictionary returned by the :py:meth:`stacie.cost.LowFreqCost.props` method.

    Returns
    -------
    results
        A dictionary with "criterion" and other fields.
        (See module docstring for details.)
    """
    amplitudes_rest = props["amplitudes_rest"]
    kappas_rest = props["kappas_rest"]
    thetas_rest = amplitudes_rest / (kappas_rest - 1)
    ll_lowfreq = props["ll"]
    ll_rest = (
        -gammaln(kappas_rest)
        - np.log(thetas_rest)
        + (kappas_rest - 1) * (np.log(kappas_rest - 1) - 1)
    ).sum()
    npar_lowfreq = len(props["pars"])
    npar_rest = len(amplitudes_rest)
    return {"criterion": 2 * (npar_lowfreq + npar_rest) - 2 * (ll_lowfreq + ll_rest)}


@mark_criterion(half_opt=True)
def halfhalf_criterion(props: dict[str, NDArray]) -> float:
    """Likelihood that the same parameters fit both the first and second halves of the spectrum.

    Parameters
    ----------
    props
        The property dictionary returned by the :py:meth:`stacie.cost.LowFreqCost.props` method.

    Returns
    -------
    results
        A dictionary with "criterion" and other fields.
        (See module docstring for details.)
    """
    # Sanity check: we need positive definite hessians.
    cost_hess_evals = props["cost_hess_evals"]
    if np.any(cost_hess_evals <= 0) or not np.all(np.isfinite(cost_hess_evals)):
        return {"criterion": np.inf}
    # Idem for the Hessians of the fits to the two halves.
    hess1 = props["cost_hess_half1"]
    hess2 = props["cost_hess_half2"]
    if not (np.isfinite(hess1).all() and np.isfinite(hess2).all()):
        return {"criterion": np.inf}
    evals1 = np.linalg.eigvalsh(hess1)
    evals2 = np.linalg.eigvalsh(hess2)
    if not ((evals1 > 0).all() and (evals2 > 0).all()):
        return {"criterion": np.inf}

    # Compute the difference in parameters and the expected covariance of this difference.
    delta = props["pars_half1"] - props["pars_half2"]
    covar = np.linalg.inv(hess1) + np.linalg.inv(hess2)
    if not np.isfinite(covar).all():
        return {"criterion": np.inf}
    evals, evecs = np.linalg.eigh(covar)
    # Again, the covariance of the difference must be positive definite.
    # This is unlikely to fail, but may occasionally happen due to rounding errors.
    if not ((evals > 0).all() and np.isfinite(evals).all()):
        return {"criterion": np.inf}
    # Transform to the eigenbasis of the covariance.
    delta = np.dot(evecs.T, delta)

    # Compute the negative likelihood of the difference in parameters.
    nll = 0.5 * (delta**2 / evals).sum() + 0.5 * np.log(2 * np.pi * evals).sum()

    # Compute the expected value of the negative likelihood, which is the entropy.
    entropy = 0.5 * len(delta) + 0.5 * np.log(2 * np.pi * evals).sum()

    return {
        "criterion": nll,
        "criterion_expected": entropy,
        "criterion_scale": 2 * len(delta),
    }


@mark_criterion()
def halfapprox_criterion(props: dict[str, NDArray]) -> float:
    """Approximate the halfhalf criterion without requiring a reoptimization.

    Parameters
    ----------
    props
        The property dictionary returned by the :py:meth:`stacie.cost.LowFreqCost.props` method.

    Returns
    -------
    results
        A dictionary with "criterion" and other fields.
        (See module docstring for details.)
    """
    # Sanity check: we need positive definite hessians.
    cost_hess_evals = props["cost_hess_evals"]
    if np.any(cost_hess_evals <= 0) or not np.all(np.isfinite(cost_hess_evals)):
        return {"criterion": np.inf}

    # Compute the gradient of the cost function for the first and second half.
    nfreq = len(props["freqs"])
    sensitivity = props["cost_grad_sensitivity"]
    residual = props["amplitudes"] - props["amplitudes_model"][0]
    grad1 = np.dot(sensitivity[:, : nfreq // 2], residual[: nfreq // 2])
    grad2 = np.dot(sensitivity[:, nfreq // 2 :], residual[nfreq // 2 :])

    # Transform the difference in gradients to the eigenbasis of the covariance.
    cost_hess_evecs = props["cost_hess_evecs"]
    delta_grad = np.dot(cost_hess_evecs.T, grad1 - grad2)

    # Approximate the difference in parameters to first-order,
    # assuming that the Hessian is twice as flat when using half of the data.
    delta_pars = 2 * delta_grad / cost_hess_evals

    # Approximate the eigenvalues of the covaraince of delta_pars:
    covar_evals_delta = 4 / cost_hess_evals
    # The factor 4 is motivated as follows:
    # - The covariance is twice as large because only half of the data is used.
    # - The covariance of the difference is the sum of the covariances of the two fits.

    # Compute the negative log likelihood of the difference in parameters.
    nll = (
        0.5 * (delta_pars**2 / covar_evals_delta).sum()
        + 0.5 * np.log(2 * np.pi * covar_evals_delta).sum()
    )

    # Compute the expected value of the negative log likelihood, which is the entropy.
    entropy = 0.5 * len(delta_pars) + 0.5 * np.log(2 * np.pi * covar_evals_delta).sum()

    return {
        "criterion": nll,
        "criterion_expected": entropy,
        "criterion_scale": 2 * len(delta_pars),
    }


@mark_criterion()
def evidence_criterion(props: dict[str, NDArray]) -> float:
    """Minus the logarithm of the evidence, in the MAP approximation, up to a constant.

    Parameters
    ----------
    props
        The property dictionary returned by the :py:meth:`stacie.cost.LowFreqCost.props` method.

    Returns
    -------
    results
        A dictionary with "criterion" and other fields.
        (See module docstring for details.)
    """
    # Sanity check: we need positive definite hessians.
    cost_hess_evals = props["cost_hess_evals"]
    if np.any(cost_hess_evals <= 0) or not np.all(np.isfinite(cost_hess_evals)):
        return {"criterion": np.inf}

    # calculate the evidence
    return {
        "criterion": -props["ll"] + 0.5 * np.log(2 * np.pi * cost_hess_evals).sum(),
        "criterion_scale": 2 * len(cost_hess_evals),
    }
