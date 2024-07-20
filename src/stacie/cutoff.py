# Stacie is a STable AutoCorrelation Integral Estimator.
# Copyright (C) 2024  Toon Verstraelen
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
from scipy.special import gammaln

__all__ = ("CutoffCriterion", "underfitting_criterion", "general_ufc", "akaike_criterion")


CutoffCriterion = NewType("CutoffCriterion", Callable[[dict[str, NDArray]], float])


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
    nres = len(residuals)
    if nres < 2:
        raise TypeError("The underfitting criterion is meaningless for zero or one residual.")
    scs = np.zeros(nres + 1)
    np.cumsum(residuals, out=scs[1:])
    scs -= scs[-1] / 2
    scs *= 2
    return (scs**2).mean() - nres


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
    kappas_rest = 0.5 * props["ndofs_rest"]
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
