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

from functools import partial, wraps

import attrs
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from numpy.typing import NDArray

from .cutobj import CutObj
from .model import SpectrumModel

__all__ = ("LowFreqCost",)


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


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

    cutobj: CutObj = attrs.field()
    """The criterion used to determine the frequency cutoff to select the low-frequency part."""

    def func(self, pars: NDArray) -> float:
        """Compute the cost function (the negative log-likelihood).

        Parameters
        ----------
        pars
            The parameters.

        Returns
        -------
        negll
            The negative log-likelihood of the parameters for the given ACF data.
        """
        if not self.model.valid(pars):
            return np.inf
        return _func(pars, *attrs.astuple(self))

    def prop(self, pars: NDArray) -> dict[str, NDArray]:
        """Compute properties of the fit for the given parameters."""
        if not self.model.valid(pars):
            raise ValueError("Invalid parameters")
        return _prop(pars, *attrs.astuple(self))

    def grad(self, pars: NDArray) -> NDArray:
        """Compute the gradient of the cost function."""
        if not self.model.valid(pars):
            return np.full(len(pars), np.nan)
        return _grad(pars, *attrs.astuple(self))

    def hess(self, pars: NDArray) -> NDArray:
        """Compute the Hessian of the cost function."""
        if not self.model.valid(pars):
            return np.full((len(pars), len(pars)), np.nan)
        return _hess(pars, *attrs.astuple(self))


def cost_low(
    pars: NDArray[float],
    timestep: float,
    freqs: NDArray[float],
    amplitudes: NDArray[float],
    ndofs: NDArray[int],
    model: SpectrumModel,
    cutobj: CutObj,
    *,
    do_props: bool = False,
) -> jax.Array | dict[str, jax.Array]:
    """Low-level implementation of the cost function.

    Parameters
    ----------
    pars
        The parameter vector for which the loss function must be computed.
    do_props
        Set to ``True`` to let this function compute a dictionary of properties.

    Returns
    -------
    If ``do_props==False``, the return value is minus the log-likelihood.
    If ``do_props==True``, this function returns a dictionary with various intermediate results
    of the loss function calculations:

    - ``pars``: the given parameters
    - ``ll``: the log likelihood
    - ``uni``: the residuals transformed such that they should be uniformely distributed.
    - ``uni``: the residuals transformed such that they should be normally distributed.
    - ``obj``: the objective to be minimized to find the best frequency cutoff.
    - ``amplitudes_model``: The model of the spectrum (function of pars)
    - ``amplitudes_std_model``: The model of the standard error of the spectrum (function of pars)

    Notes
    -----
    For all other parameters, see attributes of the ``LowFreqCost`` class.
    """
    # Convert frequencies to dimensionless omegas, as if time step was 1
    # With RFFT, the highest omega would then be +pi.
    omegas = 2 * jnp.pi * timestep * freqs

    amplitudes_model = model(omegas, pars)

    # Log-likelihood computed with the scaled Chi-squared distribution
    # - Variance of the real or imaginary component of a Fourier-transform,
    #   taking into account that it is rescaled when computing the average spectrum.
    var_ft = amplitudes_model / ndofs
    # - Transform to dimensionless, standard Chi-squared random variables
    xsq = amplitudes / var_ft
    # - Compute the log-likelihood
    ll_norm1 = -0.5 * (ndofs * jnp.log(2)).sum()
    ll_norm2 = -jsp.gammaln(0.5 * ndofs).sum()
    ll_power = ((0.5 * ndofs - 1) * jnp.log(xsq)).sum()
    ll_cor = -jnp.log(var_ft).sum()
    ll_exp = -0.5 * (xsq).sum()
    ll = ll_norm1 + ll_norm2 + ll_power + ll_cor + ll_exp

    if do_props:
        # Transformation to normal errors
        uni = jsp.gammainc(0.5 * ndofs, 0.5 * xsq)
        nor = jsp.erfinv(2 * uni - 1) * jnp.sqrt(2)
        # Objective = minimize excess variance of the cumulative sum of the normal noise.
        obj = cutobj(nor)
        return {
            "pars": pars,
            "ll": ll,
            "uni": uni,
            "nor": nor,
            "obj": obj,
            "amplitudes_model": amplitudes_model,
            "amplitudes_std_model": amplitudes_model / jnp.sqrt(0.5 * ndofs),
        }
    return -ll


def numpify(func):
    """Convert return values from JAX arrays to NumPy arrays."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, dict):
            return {key: np.array(value) for key, value in result.items()}
        return np.array(result)

    return wrapper


# Jit-compile functions to be used in LowFreqCost
STATIC_ARGNAMES = ("model", "cutobj", "do_props")
_func = numpify(jax.jit(cost_low, static_argnames=STATIC_ARGNAMES))
_prop = numpify(jax.jit(partial(cost_low, do_props=True), static_argnames=STATIC_ARGNAMES))
_grad = numpify(jax.jit(jax.grad(cost_low), static_argnames=STATIC_ARGNAMES))
_hess = numpify(jax.jit(jax.hessian(cost_low), static_argnames=STATIC_ARGNAMES))
