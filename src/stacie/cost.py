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
from numpy.typing import NDArray
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

    model: SpectrumModel = attrs.field()
    """The model to be fitted to the spectrum."""

    def __call__(self, pars: NDArray[float], deriv: int = 0) -> float:
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
        if not self.model.valid(pars):
            if deriv == 0:
                return [np.inf]
            if deriv == 1:
                return [np.inf, np.full_like(pars, np.nan)]
            if deriv == 2:
                return [np.inf, np.full_like(pars, np.nan), np.full((len(pars), len(pars)), np.nan)]
            raise ValueError("Third or higher derivatives are not supported.")
        props = cost_low(pars, deriv, *attrs.astuple(self, recurse=False))
        if deriv == 0:
            return [props["cost_value"]]
        if deriv == 1:
            return [props["cost_value"], props["cost_grad"]]
        if deriv == 2:
            return [props["cost_value"], props["cost_grad"], props["cost_hess"]]
        raise ValueError("Third or higher derivatives are not supported.")

    def props(self, pars: NDArray[float], deriv: int = 0) -> dict[str, NDArray[float]]:
        """Compute properties of the fit for the given parameters.

        Parameters
        ----------
        pars
            Parameter vector passed on to ``self.model``.
        deriv
            The maximum order of derivatives to compute: 0, 1 or 2.

        Returns
        -------
        props
            Dictionary with properties.
            See :py:func:`cost_low` for details.
        """
        if not self.model.valid(pars):
            raise ValueError("Invalid parameters")
        return cost_low(pars, deriv, *attrs.astuple(self, recurse=False), do_props=True)


def cost_low(
    pars: NDArray[float],
    deriv: int,
    freqs: NDArray[float],
    ndofs: NDArray[int],
    amplitudes: NDArray[float],
    model: SpectrumModel,
    *,
    do_props: bool = False,
) -> dict[str, NDArray]:
    """Low-level implementation of the cost function.

    Only ``pars`` and ``deriv`` parameters are documented below.
    For all other parameters, see attributes of the :class:`LowFreqCost` class.

    Parameters
    ----------
    pars
        The parameter vector for which the loss function must be computed.
    deriv
        The order of derivatives of the cost function to include.
    do_props
        Whether to include additional properties in the output.

    Returns
    -------
    props
        A dictionary with various intermediate results of the loss function calculations.
        See notes for details.

    Notes
    -----
    The returned dictionary contains the following items:

    - ``cost_value``: the cost function value.
    - ``cost_grad``: the cost Gradient vector (if ``deriv>=1``).
    - ``cost_hess``: the cost Hessian matrix (if ``deriv==2``).

    If ``do_props=True``, the dictionary also contains the following items:

    - ``pars``: the given parameters.
    - ``freqs``: the given frequencies.
    - ``amplitudes``: the given frequencies.
    - ``kappas``: shape parameters for the gamma distribution.
    - ``thetas``: scale parameters for the gamma distribution.
    - ``amplitudes``: the given frequencies.
    - ``ll``: the log likelihood.
    - ``cost_grad_sensitivity``: the sensitivity of the cost gradient to the amplitudes
      (if ``deriv==2``).
    """
    # Compute the model spectrum and its derivatives.
    amplitudes_model = model.compute(freqs, pars, deriv)
    kappas = 0.5 * ndofs
    thetas = amplitudes_model[0] / kappas
    if (amplitudes_model[0] <= 0).any():
        # Avoid warnings due to negative model amplitudes.
        ll = -np.inf
        props = {"cost_value": np.inf}
        if deriv >= 1:
            props["cost_grad"] = np.full_like(pars, np.nan)
        if deriv >= 2:
            props["cost_hess"] = np.full((len(pars), len(pars)), np.nan)
    else:
        # Log-likelihood computed with the scaled Chi-squared distribution.
        # The Gamma distribution is used because the scale parameter is easily incorporated.
        ll_terms = logpdf_gamma(amplitudes, kappas, thetas, deriv)
        ll = ll_terms[0].sum()

        props = {
            "cost_value": -ll,
        }
        if deriv >= 1:
            props["cost_grad"] = -np.dot(amplitudes_model[1], ll_terms[1] / kappas)
        if deriv >= 2:
            props["cost_hess"] = -(
                np.einsum(
                    "ia,ja,a->ij", amplitudes_model[1], amplitudes_model[1], ll_terms[2] / kappas**2
                )
                + np.einsum("ija,a->ij", amplitudes_model[2], ll_terms[1] / kappas)
            )
        if deriv >= 3:
            raise ValueError("Third or higher derivatives are not supported.")

    if do_props:
        props.update(
            {
                "pars": pars,
                "freqs": freqs,
                "amplitudes": amplitudes,
                "kappas": kappas,
                "thetas": thetas,
                "amplitudes_model": amplitudes_model,
                "ll": ll,
                "entropy": entropy_gamma(kappas, thetas[0])[0].sum() if np.isfinite(ll) else np.inf,
            }
        )
        if deriv >= 2:
            # Mix derivative of the cost with respect to
            # (i) the model parameters and (ii) the empiricl spectrum.
            props["cost_grad_sensitivity"] = -(amplitudes_model[1] / thetas**2 / kappas)

    return props


def logpdf_gamma(x: NDArray[float], kappa: NDArray[float], theta: NDArray[float], deriv: int = 1):
    """Compute the logarithm of the probability density function of the Gamma distribution.

    Parameters
    ----------
    x
        The argument of the PDF (random variable).
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
