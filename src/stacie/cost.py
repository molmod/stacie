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
"""Cost function to optimize models for the low-frequency part of the spectrum."""

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, gammaln, polygamma

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

    def __call__(self, pars: NDArray[float], *, deriv: int = 0) -> float:
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
            self.freqs, pars if mask.ndim == 0 else pars[mask], deriv=deriv
        )
        alphas = 0.5 * self.ndofs

        # Only continue with parameters for which the model does not become negative.
        # Small positive values are also excluded to avoid underflows.
        pos_mask = (amplitudes_model[0] > 1e-30).all(axis=-1)
        if not pos_mask.any():
            return results
        if mask.ndim > 0:
            amplitudes_model = [am[pos_mask] for am in amplitudes_model]
            mask[mask] = pos_mask
        del pos_mask

        # Log-likelihood computed with the scaled Chi-squared distribution.
        # The Gamma distribution is used because the scale parameter is easily incorporated.
        thetas = [am / alphas for am in amplitudes_model]
        ll_terms = logpdf_gamma(self.amplitudes, alphas, thetas[0], deriv=deriv)
        nlp = self.model.neglog_prior(pars[mask] if mask.ndim > 0 else pars, deriv=deriv)
        results[0][mask] = -np.einsum("...i,i->...", ll_terms[0], self.weights) + nlp[0]
        if deriv >= 1:
            results[1][mask] = (
                -np.einsum("...pi,...i,i->...p", thetas[1], ll_terms[1], self.weights) + nlp[1]
            )
        if deriv >= 2:
            results[2] = (
                -(
                    np.einsum(
                        "...pi,...qi,...i,i->...pq", thetas[1], thetas[1], ll_terms[2], self.weights
                    )
                    + np.einsum("...pqi,...i,i->...pq", thetas[2], ll_terms[1], self.weights)
                )
                + nlp[2]
            )
        return results

    def expected(self, pars: NDArray[float]) -> NDArray[float]:
        """Compute the expected value and variance of the cost function.

        Parameters
        ----------
        pars
            The model parameters.
            Vectorization is not supported yet.

        Returns
        -------
        expected, variance
            The expected value and variance of the cost function.
        """
        pars = np.asarray(pars)
        amplitudes_model = self.model.compute(self.freqs, pars)
        alphas = 0.5 * self.ndofs
        thetas = amplitudes_model[0] / alphas
        return (
            np.dot(entropy_gamma(alphas, thetas)[0], self.weights),
            np.dot(varlogp_gamma(alphas), self.weights**2),
        )


def logpdf_gamma(
    x: NDArray[float], alpha: NDArray[float], theta: NDArray[float], *, deriv: int = 0
):
    """Compute the logarithm of the probability density function of the Gamma distribution.

    Parameters
    ----------
    x
        The argument of the PDF (random variable).
        Array with shape ``(nfreq,)``.
    alpha
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
    alpha = np.asarray(alpha)
    theta = np.asarray(theta)
    ratio = np.asarray(x) / theta
    results = [-gammaln(alpha) - np.log(theta) + (alpha - 1) * np.log(ratio) - ratio]
    if deriv >= 1:
        results.append((ratio - alpha) / theta)
    if deriv >= 2:
        results.append((alpha - 2 * ratio) / theta**2)
    if deriv >= 3:
        raise ValueError("Third or higher derivatives are not supported.")
    return results


def entropy_gamma(alpha: NDArray[float], theta: NDArray[float], *, deriv: int = 0):
    """Compute the entropy of the Gamma distribution.

    Parameters
    ----------
    alpha
        The shape parameter.
    theta
        The scale parameter.
    deriv
        The order of the derivatives toward theta to compute: 0, 1 or 2.

    Returns
    -------
    results
        A list of results (function value and requested derivatives.)
        All elements have the same shape as the ``alpha`` and ``theta`` arrays.
    """
    alpha = np.asarray(alpha)
    theta = np.asarray(theta)
    results = [alpha + np.log(theta) + gammaln(alpha) + (1 - alpha) * digamma(alpha)]
    if deriv >= 1:
        results.append(1 / theta)
    if deriv >= 2:
        results.append(-1 / theta**2)
    if deriv >= 3:
        raise ValueError("Third or higher derivatives are not supported.")
    return results


def varlogp_gamma(alpha: NDArray[float]) -> NDArray[float]:
    """Compute the variance of the log-probability density function of the Gamma distribution.

    Parameters
    ----------
    alpha
        The shape parameter.

    Returns
    -------
    var
        The variance of the log-probability density function.
        Array with shape ``(alpha,)``.
    """
    alpha = np.asarray(alpha)
    return (alpha - 1) ** 2 * polygamma(1, alpha) - alpha + 2
