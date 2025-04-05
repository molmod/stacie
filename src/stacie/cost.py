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
"""Cost function to optimize models for the low-frequency part of the spectrum."""

import attrs
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import digamma, gammaln

from .model import SpectrumModel

__all__ = ("LowFreqCost", "entropy_gamma", "logpdf_gamma")


@attrs.define
class LowFreqCost:
    """Cost function to fit a model to the low-frequency part of the spectrum."""

    freqs: NDArray[float] = attrs.field()
    """The frequencies for which the spectrum amplitudes are computed."""

    ndofs: NDArray[int] = attrs.field()
    """The number of independent contributions to each spectrum amplitude."""

    amplitudes: NDArray[float] = attrs.field()
    """The actual spectrum amplitudes at frequencies in ``self.freqs``."""

    weights: NDArray[float] = attrs.field()
    """The fitting weights for each grid point."""

    model: SpectrumModel = attrs.field()
    """The model to be fitted to the spectrum."""

    def __call__(self, pars: ArrayLike, deriv: int = 0) -> float:
        """Evaluate the cost function and its derivatives.

        Parameters
        ----------
        pars
            The parameter vector for which the loss function must be computed.
        deriv
            The order of derivatives of the cost function to include.

        Returns
        -------
        results
            A list with the cost function and the requested derivatives.
        """
        # Prepare result arrays, with inf and nan by default.
        # These will be overwritten if the model is valid.
        pars = np.asarray(pars)
        vec_shape = pars.shape[:-1]
        par_shape = pars.shape[-1:]
        results = [np.full(vec_shape, np.inf)]
        if deriv >= 1:
            results.append(np.full(vec_shape + par_shape, np.nan))
        if deriv >= 2:
            results.append(np.full(vec_shape + par_shape + par_shape, np.nan))
        if deriv >= 3:
            raise ValueError("Third or higher derivatives are not supported.")

        mask = self.model.valid(pars)
        if not mask.any():
            return results

        # Compute the model spectrum and its derivatives.
        amplitudes_model = self.model.compute(
            self.freqs, pars if mask.ndim == 0 else pars[mask], deriv
        )
        kappas = 0.5 * self.ndofs

        # Only continue with parameters for which the model does not become negative.
        pos_mask = (amplitudes_model[0] > 0).all(axis=-1)
        if not pos_mask.any():
            return results
        if mask.ndim > 0:
            amplitudes_model = [am[pos_mask] for am in amplitudes_model]
            mask[mask] = pos_mask
        del pos_mask

        # Log-likelihood computed with the scaled Chi-squared distribution.
        # The Gamma distribution is used because the scale parameter is easily incorporated.
        thetas = [am / kappas for am in amplitudes_model]
        ll_terms = logpdf_gamma(self.amplitudes, kappas, thetas[0], deriv)
        results[0][mask] = -np.einsum("...i,i->...", ll_terms[0], self.weights)
        if deriv >= 1:
            results[1][mask] = -np.einsum(
                "...pi,...i,i->...p", thetas[1], ll_terms[1], self.weights
            )
        if deriv >= 2:
            results[2] = -(
                np.einsum(
                    "...pi,...qi,...i,i->...pq", thetas[1], thetas[1], ll_terms[2], self.weights
                )
                + np.einsum("...pqi,...i,i->...pq", thetas[2], ll_terms[1], self.weights)
            )
        return results


def logpdf_gamma(x: NDArray[float], kappa: NDArray[float], theta: NDArray[float], deriv: int = 1):
    """Compute the logarithm of the probability density function of the Gamma distribution.

    Parameters
    ----------
    x
        The argument of the PDF (random variable).
        Array with shape ``(nfreq,)``.
    kappa
        The shape parameter.
        Array with shape ``(nfreq,)``.
    theta
        The scale parameter.
        Array with shape ``(..., nfreq,)``.
    deriv
        The order of the derivatives toward theta to compute: 0, 1 or 2.

    Returns
    -------
    results
        A list of results (function value and requested derivatives.)
        All elements have the same shape as the ``theta`` array.
    """
    kappa = np.asarray(kappa)
    theta = np.asarray(theta)
    ratio = np.asarray(x) / theta
    results = [-gammaln(kappa) - np.log(theta) + (kappa - 1) * np.log(ratio) - ratio]
    if deriv >= 1:
        results.append((ratio - kappa) / theta)
    if deriv >= 2:
        results.append((kappa - 2 * ratio) / theta**2)
    if deriv >= 3:
        raise ValueError("Third or higher derivatives are not supported.")
    return results


def entropy_gamma(kappa: NDArray[float], theta: NDArray[float], deriv: int = 1):
    """Compute the entropy of the Gamma distribution.

    Parameters
    ----------
    kappa
        The shape parameter.
    theta
        The scale parameter.
    deriv
        The order of the derivatives toward theta to compute: 0, 1 or 2.

    Returns
    -------
    results
        A list of results (function value and requested derivatives.)
        All elements have the same shape as the ``kappa`` and ``theta`` arrays.
    """
    kappa = np.asarray(kappa)
    theta = np.asarray(theta)
    results = [kappa + np.log(theta) + gammaln(kappa) + (1 - kappa) * digamma(kappa)]
    if deriv >= 1:
        results.append(1 / theta)
    if deriv >= 2:
        results.append(-1 / theta**2)
    if deriv >= 3:
        raise ValueError("Third or higher derivatives are not supported.")
    return results
