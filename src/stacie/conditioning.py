# STACIE is a STable AutoCorrelation Integral Estimator.
# Copyright (C) 2024-2025 The contributors of the STACIE Python Package.
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
"""Cost function pre-conditioning."""

from collections.abc import Callable

import attrs
import numpy as np
from numpy.typing import NDArray

__all__ = ("ConditionedCost",)


@attrs.define
class ConditionedCost:
    """A wrapper for the cost function taking care of pre-conditioning.

    The goal of the pre-conditioner is to let the optimizer work with normalized parameters,
    and to scale the cost function to a normalized range, such that all quantities are close to 1,
    even if the spectra and the frequencies have very different orders of magnitude.
    """

    cost: Callable[[NDArray[float], int], list[NDArray[float]]] = attrs.field()
    par_scales: NDArray[float] = attrs.field()
    cost_scale: float = attrs.field()

    def __call__(self, pars: NDArray[float], *, deriv: int = 0) -> list[NDArray[float]]:
        """Evaluate the pre-conditioned cost function.

        Parameters
        ----------
        pars
            The parameters to evaluate the cost function at, in the original space.
            For vectorized calculations, use N-dimensional inputs of which the last axis
            corresponds to the parameters.
        deriv
            The order of the derivative to compute.

        Returns
        -------
        results
            The cost function value and its derivatives.
            In vectorized calculations, the last axis of the gradient
            and the last two of the Hessian correspond to the parameters.
        """
        if not isinstance(deriv, int):
            raise TypeError("Argument deriv must be integer.")
        if deriv < 0:
            raise ValueError("Argument deriv must be zero or positive.")
        pars_orig = pars * self.par_scales
        results_orig = self.cost(pars_orig, deriv=deriv)
        results = [results_orig[0] / self.cost_scale]
        if deriv == 0:
            return results
        results.append(results_orig[1] * (self.par_scales / self.cost_scale))
        if deriv == 1:
            return results
        results.append(
            results_orig[2] * (np.outer(self.par_scales, self.par_scales) / self.cost_scale)
        )
        if deriv == 2:
            return results
        raise NotImplementedError("Third and higher-order derivatives are not supported.")

    def to_reduced(self, pars: NDArray[float]) -> NDArray[float]:
        """Convert parameters from the original to the reduced space.

        Parameters
        ----------
        pars
            The parameters to convert, in the original space.

        Returns
        -------
        pars_reduced
            The parameters in the reduced space.
        """
        return pars / self.par_scales

    def from_reduced(self, pars: NDArray[float]) -> NDArray[float]:
        """Convert parameters from the reduced to the original space.

        Parameters
        ----------
        pars
            The parameters to convert, in the reduced space.

        Returns
        -------
        pars_orig
            The parameters in the original space.
        """
        return pars * self.par_scales

    def funcgrad(self, pars: NDArray[float]) -> tuple[float, NDArray[float]]:
        """Compute the cost function and the gradient.

        Parameters
        ----------
        pars
            The parameters, in the reduced space.

        Returns
        -------
        cost_reduced
            The cost normalized function value.
        """
        return self(pars, deriv=1)

    def hess(self, pars: NDArray[float]) -> NDArray[float]:
        """Compute the Hessian matrix of the cost function."""
        return self(pars, deriv=2)[2]
