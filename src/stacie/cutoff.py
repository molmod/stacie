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

from .utils import PositiveDefiniteError, robust_posinv

__all__ = (
    "CutoffCriterion",
    "akaike_criterion",
    "cv2_criterion",
    "cv2l_criterion",
    "entropy_criterion",
    "expected_ufc",
    "general_ufc",
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
def entropy_criterion(props: dict[str, np.ndarray]) -> dict[str, float]:
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
        "criterion_scale": nlwe_expected,
    }


@mark_criterion()
def underfitting_criterion(props: dict[str, NDArray]) -> dict[str, float]:
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
        The array should have a shape of ``(nparameter, nfreq)``, where each row
        represents a basis vector. The residuals must be orthogonal to these basis
        functions. Note that the basis vectors do not need to be orthogonal or normalized.

    Returns
    -------
    expected
        The expected value of the underfitting criterion
        (averaged over all possible residuals orthogonal to the basis functions).
    """
    # Construct the orthonormal basis of derivatives of the model w.r.t. parameters.
    nbasis, nfreq = basis.shape
    basis /= np.linalg.norm(basis, axis=1).reshape(-1, 1)
    basis = np.linalg.svd(basis, full_matrices=False)[2]

    # 'scs' is the abbreviation of 'symmetric cumulative sum'.
    scs_basis = np.zeros((nbasis, nfreq + 1))
    np.cumsum(basis, out=scs_basis[:, 1:], axis=1)
    scs_basis -= scs_basis[:, -1:] / 2
    scs_basis *= 2
    uvec_sqmodel = nfreq - np.sum(scs_basis**2, axis=0)
    return uvec_sqmodel.sum() / (nfreq + 1) - nfreq


@mark_criterion()
def sumsq_criterion(props: dict[str, np.ndarray]) -> dict[str, float]:
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
def akaike_criterion(props: dict[str, NDArray]) -> dict[str, float]:
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
def cv2_criterion(props: dict[str, NDArray], precondition: bool = True) -> dict[str, float]:
    """Likelihood that the same parameters fit both the first and second halves of the spectrum.

    This is a form of cross-validation by fitting the same parameters to two halves of the spectrum.
    The "risk" of a generalization error is gauged by the negative log likelihood
    of the difference between the two fitted parameter vectors.

    Parameters
    ----------
    props
        The property dictionary returned by the :py:meth:`stacie.cost.LowFreqCost.props` method.
    precondition
        Set to False to disable preconditioning of the Hessian eigendecomposition.
        This is only used for testing.

    Returns
    -------
    results
        A dictionary with "criterion" and other fields.
        (See module docstring for details.)
    """
    # Sanity check: we need positive definite hessians.
    rescaled_evals = props["cost_hess_rescaled_evals"]
    if np.any(rescaled_evals <= 0) or not np.all(np.isfinite(rescaled_evals)):
        return {
            "criterion": np.inf,
            "criterion_error": "Hessian of full fit is not positive definite.",
        }
    # Idem for the Hessians of the fits to the two halves.
    hess1 = props["cost_hess_half1"]
    hess2 = props["cost_hess_half2"]
    if not (np.isfinite(hess1).all() and np.isfinite(hess2).all()):
        return {"criterion": np.inf, "criterion_error": "Hessians of half fits are not finite."}

    # Condition the hessians with the hess_scales of the full fit.
    hess_scales = props["cost_hess_scales"] if precondition else np.ones(len(hess1))
    hess1 = (hess1 / hess_scales) / hess_scales[:, None]
    hess2 = (hess2 / hess_scales) / hess_scales[:, None]
    # Sanity check on the separate Hessians.
    evals1 = np.linalg.eigvalsh(hess1)
    evals2 = np.linalg.eigvalsh(hess2)
    if not ((evals1 > 0).all() and (evals2 > 0).all()):
        return {
            "criterion": np.inf,
            "criterion_error": "Hessians of half fits are not positive definite.",
        }

    # Compute the difference in parameters and the expected covariance of this difference
    # in rescaled coordinates.
    delta = (props["pars_half1"] - props["pars_half2"]) * hess_scales
    covar = np.linalg.inv(hess1) + np.linalg.inv(hess2)
    if not np.isfinite(covar).all():
        return {
            "criterion": np.inf,
            "criterion_error": "Covariance of parameter difference is not finite.",
        }
    evals, evecs = np.linalg.eigh(covar)
    # Again, the covariance of the difference must be positive definite.
    # This is unlikely to fail, but may occasionally happen due to rounding errors.
    if not ((evals > 0).all() and np.isfinite(evals).all()):
        return {
            "criterion": np.inf,
            "criterion_error": "Covariance of parameter difference is not positive definite.",
        }
    # Transform to the eigenbasis of the covariance.
    delta = np.dot(evecs.T, delta)

    # Compute the negative likelihood of the difference in parameters.
    nll = (
        0.5 * (delta**2 / evals).sum()
        + 0.5 * np.log(2 * np.pi * evals).sum()
        - np.log(hess_scales).sum()
    )

    # Compute the expected value of the negative likelihood, which is the entropy.
    entropy = 0.5 * len(delta) + 0.5 * np.log(2 * np.pi * evals).sum() - np.log(hess_scales).sum()

    return {
        "criterion": nll,
        "criterion_expected": entropy,
        "criterion_scale": len(delta),
    }


@mark_criterion()
def cv2l_criterion(
    props: dict[str, NDArray], precondition: bool = True, convergence_check: bool = False
) -> dict[str, float]:
    """Linearly approximate cv2_criterion without requiring a reoptimization.

    Parameters
    ----------
    props
        The property dictionary returned by the :py:meth:`stacie.cost.LowFreqCost.props` method.
    precondition
        Set to False to disable preconditioning of the covariance eigendecomposition.
        This is only used for testing.
    convergence_check
        Add an empirical penalty term to the criterion that is sensitive to poor convergence
        of the fit to the whole spectrum.
        This is only useful for debugging.

    Returns
    -------
    results
        A dictionary with "criterion" and other fields.
        (See module docstring for details.)
    """
    # Sanity checks
    npoint = len(props["amplitudes"])
    if npoint % 2 != 0:
        raise ValueError(f"The number of points in the regression must be even, got {npoint}.")
    nhalf = npoint // 2

    # Construct a linear regression approximation with normally distributed errors,
    # of the original non-linear problem with Gamma-distributed errors.
    design_matrix = props["amplitudes_model"][1].T
    expected_values = props["amplitudes"] - props["amplitudes_model"][0]
    # Transform equations to a basis with standard normal errors.
    data_std = np.sqrt(props["thetas"] ** 2 * props["kappas"])
    design_matrix = design_matrix / data_std.reshape(-1, 1)
    expected_values = expected_values / data_std

    # Precondition the basis.
    u, s, vt = np.linalg.svd(design_matrix, full_matrices=False)
    # Solve the preconditioned problem for the first and second half with SVD.
    u1, s1, vt1 = np.linalg.svd(u[:nhalf], full_matrices=False)
    u2, s2, vt2 = np.linalg.svd(u[nhalf:], full_matrices=False)
    # Solutions of the preconditioned problem
    px1 = np.dot(vt1.T, np.dot(u1.T, expected_values[:nhalf]) / s1)
    px2 = np.dot(vt2.T, np.dot(u2.T, expected_values[nhalf:]) / s2)
    # Covariance matrices of the preconditioned problem
    pc1 = np.einsum("ji,j,jk", vt1, s1**-2, vt1)
    pc2 = np.einsum("ji,j,jk", vt2, s2**-2, vt2)

    # Transform back to the original parameter space.
    x1 = np.dot(vt.T, px1 / s)
    x2 = np.dot(vt.T, px2 / s)
    c1 = np.einsum("ai,a,ab,b,bj", vt, 1 / s, pc1, 1 / s, vt)
    c2 = np.einsum("ai,a,ab,b,bj", vt, 1 / s, pc2, 1 / s, vt)

    # Compute the difference between the two parameter vectors in the
    # basis of the covariance matrix of the difference
    if precondition:
        try:
            scales, evals, evecs, _ = robust_posinv(c1 + c2)
        except PositiveDefiniteError:
            return {
                "criterion": np.inf,
                "criterion_error": "Covariance of parameter difference is not positive definite.",
            }
    else:
        if not (np.isfinite(c1).all() and np.isfinite(c2).all()):
            return {
                "criterion": np.inf,
                "criterion_error": "Covariance of parameter difference is not finite.",
            }
        evals, evecs = np.linalg.eigh(c1 + c2)
        if evals.min() <= 0:
            return {
                "criterion": np.inf,
                "criterion_error": "Covariance of parameter difference is not positive definite.",
            }
        scales = np.ones(len(evals))
    delta = np.dot(evecs.T, (x1 - x2) / scales)

    # Compute the negative log likelihood of the difference in parameters.
    nll = (
        0.5 * (delta**2 / evals).sum()
        + 0.5 * np.log(2 * np.pi * evals).sum()
        + np.log(scales).sum()
    )

    # Compute the expected value of the negative log likelihood, which is the entropy.
    entropy = 0.5 * len(delta) + 0.5 * np.log(2 * np.pi * evals).sum() + np.log(scales).sum()

    if convergence_check:
        # The following terms are not needed for the criterion, but may be useful for debugging.
        # They are sensitive to potential convergence issues in the optimization.
        trend = np.dot(evecs.T, (x1 + x2) / scales)
        nll += (
            0.5 * (trend**2 / evals).sum()
            + 0.5 * np.log(2 * np.pi * evals).sum()
            + np.log(scales).sum()
        )
        entropy *= 2

    return {
        "criterion": nll,
        "criterion_expected": entropy,
        "criterion_scale": len(delta),
    }


@mark_criterion()
def evidence_criterion(props: dict[str, NDArray]) -> dict[str, float]:
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
    evals = props["cost_hess_rescaled_evals"]
    scales = props["cost_hess_scales"]
    if np.any(evals <= 0) or not np.all(np.isfinite(evals)):
        return {
            "criterion": np.inf,
            "criterion_error": "Hessian of full fit is not positive definite.",
        }

    # calculate the evidence
    return {
        "criterion": -props["ll"] + 0.5 * np.log(2 * np.pi * evals).sum() + np.log(scales).sum(),
        "criterion_scale": len(evals),
    }
