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
"""Criteria for selecting the part of the spectrum to fit to."""

from collections.abc import Callable
from typing import NewType

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, gammaln

__all__ = (
    "CutoffCriterion",
    "akaike_criterion",
    "entropy_criterion",
    "general_ufc",
    "underfitting_criterion",
)


CutoffCriterion = NewType("CutoffCriterion", Callable[[dict[str, NDArray]], float])


def entropy_criterion(props: dict[str, np.ndarray]) -> float:
    r"""
    Compute the entropy criterion based on the negative log Wiener entropy (NLWE).
    In this case, the NLWE is computed using the following formula:

    .. math::

        \text{NLWE} = -\ln(\text{WE}) = \ln(\text{AM}) - \ln(\text{GM})

    where AM and GM are the arithmetic and geometric mean, respectively,
    of the spectrum amplitudes divided by the model of the spectrum.

    The expectation value of the NLWE can be derived using the properties of the Gamma distribution:

    .. math::

        \mathrm{E}[\text{NLWE}] = \psi(n \kappa) - \phi(\kappa) - \ln(n)

    where :math:`\psi` is the digamma function, :math:`n` is the number of frequencies
    and :math:`\kappa` is the shape parameter of the Gamma distribution.
    The entropy criterion is then the squared difference between the empirical and expected NLWE.

    Parameters
    ----------
    props
        The property dictionary returned by the :py:meth:`stacie.cost.LowFreqCost.props` method.

    Returns
    -------
    criterion
        The entropy criterion. Lower is better.
    """
    ratio = props["amplitudes"] / props["amplitudes_model"][0]
    nfreq = len(props["freqs"])
    kappa = props["kappas"]

    nlwe_empirical = np.log(ratio.mean()) - np.log(ratio).mean()
    nlwe_expected = (digamma(nfreq * kappa) - digamma(kappa)).mean() - np.log(nfreq)

    return (nlwe_empirical - nlwe_expected) * nfreq


def underfitting_criterion(props: dict[str, NDArray]) -> float:
    """Quantify the degree of underfitting of a smooth spectrum model to noisy data.

    Parameters
    ----------
    props
        The property dictionary returned by the :py:meth:`stacie.cost.LowFreqCost.props` method.

    Returns
    -------
    criterion
        A criterion quantifying the degree of underfitting.
        This can be used to compare different selections of (contiguous) fitting data
        to which the same model is fitted.
        Lower is better.
    """
    residuals = (props["amplitudes"] / props["thetas"] - props["kappas"]) / np.sqrt(props["kappas"])
    return general_ufc(residuals)


def general_ufc(residuals: NDArray[float]) -> float:
    """A general-purpose implementation of the underfitting criterion.

    Parameters
    ----------
    residuals
        Normalized residuals,
        i.e. with the maximum likelihood estimate of the mean and standard deviation
        of the predication at each point.

    Returns
    -------
    criterion
        A criterion quantifying the degree of underfitting.
        This can be used to compare different selections of (contiguous) fitting data
        to which the same model is fitted.
        Lower is better.
    """
    if residuals.ndim != 1:
        raise TypeError("The residuals must be given in a 1D array.")
    nfit = len(residuals)
    if nfit < 2:
        raise TypeError("The underfitting criterion is meaningless for zero or one residual.")
    scs = np.zeros(nfit + 1)
    np.cumsum(residuals, out=scs[1:])
    scs -= scs[-1] / 2
    scs *= 2
    return (scs**2).mean() - nfit


def akaike_criterion(props: dict[str, NDArray]) -> float:
    """Compute the Akaike Information Criterion (AIC) for the whole spectrum: fitted + discarded.

    Parameters
    ----------
    props
        The property dictionary returned by the :py:meth:`stacie.cost.LowFreqCost.props` method.

    Returns
    -------
    criterion
        The total AIC.
        This can be used to compare different spectrum cutoffs.
        Lower is better.
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
    return 2 * (npar_lowfreq + npar_rest) - 2 * (ll_lowfreq + ll_rest)
