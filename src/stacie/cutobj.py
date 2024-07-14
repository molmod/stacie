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

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike as JArrayLike

__all__ = ("CutObj", "cutobj_symcu")


CutObj = NewType("CutObj", Callable[[JArrayLike, JArrayLike, JArrayLike], jax.Array])


def cutobj_symcu(amplitudes: JArrayLike, kappas: JArrayLike, thetas: JArrayLike) -> jax.Array:
    """Compute the excess variance of the cumulative sum of the normal errors, starting from middle.

    Parameters
    ----------
    amplitudes
        Spectrum amplitudes.
    kappas
        The Gamma shape parameters.
    thetas
        The Gamma scale parameters.

    Returns
    -------
    obj
        The objective to minimize.
    """
    # Transformation to normal errors
    uni = jax.scipy.stats.gamma.cdf(amplitudes, kappas, scale=thetas)
    nor = jax.scipy.special.erfinv(2 * uni - 1) * jnp.sqrt(2)
    # Objective = minimize excess variance of the cumulative sum of the normal noise.
    return ((jnp.cumsum(nor) - jnp.sum(nor) / 2) ** 2).mean() - len(nor) / 4
