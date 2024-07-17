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
"""Objective functions used to decide which part of the spectrum to fit to."""

from collections.abc import Callable
from typing import NewType

import numpy as np
from numpy.typing import NDArray

__all__ = ("RiskMetric", "risk_metric_cumsum")


RiskMetric = NewType("RiskMetric", Callable[[NDArray[float]], float])


def risk_metric_cumsum(residuals: NDArray[float]) -> float:
    """Quantify the over- and underfitting risk of a smooth model to noisy data.

    Parameters
    ----------
    residuals
        Normalized residuals (zero mean and unit variance in the ideal scenario).

    Returns
    -------
    risk_metric
        A metric quantifying the risk of over- or underfitting.
        This can be used to compare different selections of (contiguous) fitting data
        to which the same model is fitted.
    """
    if residuals.ndim != 1:
        raise TypeError("The residuals must be given in a 1D array.")
    nres = len(residuals)
    if nres < 2:
        raise TypeError("The risk metric is meaningless for zero or one residual.")
    scs = np.zeros(nres + 1)
    np.cumsum(residuals, out=scs[1:])
    scs -= scs[-1] / 2
    scs *= 2
    return (scs**2).mean() - nres
