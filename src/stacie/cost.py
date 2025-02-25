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
"""Cost function to optimize models for the low-frequency part of the spectrum."""

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from .model import SpectrumModel

__all__ = ("LowFreqCost",)


@attrs.define
class LowFreqCost:
    """Cost function to fit a model to the low-frequency part of the spectrum."""

    timestep: float = attrs.field()
    """The timestep of the sequences used to compute the spectrum."""

    freqs: NDArray[float] = attrs.field()
    """The frequencies for which the spectrum amplitudes are computed."""

    amplitudes: NDArray[float] = attrs.field()
    """The actual spectrum amplitudes at frequencies in ``self.freqs``."""

    ndofs: NDArray[int] = attrs.field()
    """The number of independent contributions to each spectrum amplitude."""

    model: SpectrumModel = attrs.field()
    """The model to be fitted to the spectrum."""

    def funcgrad(self, pars: NDArray[float]) -> tuple[float, NDArray[float]]:
        """Compute the cost function (the negative log-likelihood) and the gradient.

        Parameters
        ----------
        pars
            The parameters.

        Returns
        -------
        negll
            The negative log-likelihood of the parameters.
        """
        if not self.model.valid(pars):
            return np.inf, np.full(len(pars), np.inf)
        props = cost_low(pars, 1, *attrs.astuple(self, recurse=False))
        return props["cost_value"], props["cost_grad"]

    def hess(self, pars: NDArray[float]) -> NDArray[float]:
        """Compute the Hessian matrix of the cost function."""
        if not self.model.valid(pars):
            return np.full((len(pars), len(pars)), np.inf)
        props = cost_low(pars, 2, *attrs.astuple(self, recurse=False))
        return props["cost_hess"]

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
        return cost_low(pars, deriv, *attrs.astuple(self, recurse=False))


def cost_low(
    pars: NDArray[float],
    deriv: int,
    timestep: float,
    freqs: NDArray[float],
    amplitudes: NDArray[float],
    ndofs: NDArray[int],
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

    Returns
    -------
    props
        A dictionary with various intermediate results of the loss function calculations.
        See notes for details.

    Notes
    -----
    The returned dictionary contains the following items:

    - ``pars``: the given parameters.
    - ``timestep``: the given time step.
    - ``freqs``: the given frequencies.
    - ``amplitudes``: the given frequencies.
    - ``kappas``: shape parameters for the gamma distribution.
    - ``thetas``: scale parameters for the gamma distribution.
    - ``amplitudes``: the given frequencies.
    - ``ll``: the log likelihood.
    - ``cost_value``: the cost function value.
    - ``cost_grad``: the cost Gradient vector (if ``deriv>=1``).
    - ``cost_hess``: the cost Hessian matrix (if ``deriv==2``).
    """
    # Convert frequencies to dimensionless omegas, as if time step was 1
    # With RFFT, the highest omega would then be +pi.
    omegas = 2 * np.pi * timestep * freqs

    amplitudes_model = [row / model.amplitude_scale for row in model.compute(omegas, pars, deriv)]

    # Log-likelihood computed with the scaled Chi-squared distribution.
    # The Gamma distribution is used because the scale parameter is easily incorporated.
    kappas = 0.5 * ndofs
    thetas = amplitudes_model[0] / kappas
    ll_terms = logpdf_gamma(amplitudes / model.amplitude_scale, kappas, thetas, deriv)
    ll = ll_terms[0].sum()

    props = {
        "pars": pars,
        "timestep": timestep,
        "freqs": freqs,
        "amplitudes": amplitudes,
        "kappas": kappas,
        "thetas": thetas * model.amplitude_scale,
        "ll": ll,
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
