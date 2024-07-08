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
"""Model to fit the low-frequency part of the spectrum."""

from functools import partial, wraps

import attrs
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from numpy.typing import NDArray

__all__ = ("LowFreqCost",)


@attrs.define
class LowFreqCost:
    timestep: float = attrs.field()
    freqs: NDArray[float] = attrs.field()
    amplitudes: NDArray[float] = attrs.field()
    ndofs: NDArray[int] = attrs.field()

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [
            (0, np.inf),
            (0, np.inf),
            (0, np.inf),
        ]

    def valid(self, pars):
        return all(pmin < par < pmax for (pmin, pmax), par in zip(self.bounds, pars, strict=True))

    @property
    def guess(self):
        acfint_coarse = self.amplitudes[0]
        tau = 2 / self.freqs[-1]
        return np.array([acfint_coarse / 2, acfint_coarse / 2, tau])

    def func(self, pars: NDArray) -> float:
        """Compute the negative log-likelihood.

        Parameters
        ----------
        pars
            The parameters.

        Returns
        -------
        negll
            The negative log-likelihood of the parameters for the given ACF data.
        """
        if not self.valid(pars):
            return np.inf
        return _func(pars, self.timestep, self.freqs, self.amplitudes, self.ndofs)

    def prop(self, pars: NDArray) -> dict[str, NDArray]:
        if not self.valid(pars):
            raise ValueError("Invalid parameters")
        return _prop(pars, self.timestep, self.freqs, self.amplitudes, self.ndofs)

    def grad(self, pars: NDArray) -> NDArray:
        if not self.valid(pars):
            return np.full(len(pars), np.nan)
        return _grad(pars, self.timestep, self.freqs, self.amplitudes, self.ndofs)

    def hess(self, pars: NDArray) -> NDArray:
        if not self.valid(pars):
            return np.full((len(pars), len(pars)), np.nan)
        return _hess(pars, self.timestep, self.freqs, self.amplitudes, self.ndofs)


def loss_base(
    pars: NDArray[float],
    timestep: float,
    freqs: NDArray[float],
    amplitudes: NDArray[float],
    ndofs: NDArray[int],
    *,
    do_props: bool = False,
):
    # Convert frequencies to dimensionless omegas, as if time step was 1
    # With RFFT, the highest omega would then be +pi.
    omegas = 2 * jnp.pi * timestep * freqs

    acfint_short, acfint_tail, corrtime_tail = pars
    cosines = 2 * jnp.cos(omegas)
    ratio = jnp.exp(-2 / corrtime_tail)
    tail_model = (2 - ratio * cosines) / (1 + ratio**2 - ratio * cosines)
    # tail_model = 1 / ((2/corrtime_tail)**2 + omegas**2)
    tail_model /= tail_model[0]
    spectrum_model = acfint_tail * tail_model + acfint_short

    # Log-likelihood computed with the scaled Chi-squared distribution
    # - Variance of the real or imaginary component of a Fourier-transform,
    #   taking into account that it is rescaled when computing the average spectrum.
    var_ft = spectrum_model / ndofs
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
        # Objective = minimize excess average cumulative sum of the normal noise
        # TODO: normalize on len(uni)?
        obj = ((jnp.cumsum(nor) - jnp.sum(nor) / 2) ** 2).mean() - len(uni) / 4
        return {
            "pars": pars,
            "ll": ll,
            "uni": uni,
            "nor": nor,
            "obj": obj,
            "omegas": omegas,
            "spectrum_model": spectrum_model,
        }
    return -ll


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def numpify(func):
    """Convert return values from JAX tensors to NumPy arrays."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, dict):
            return {key: np.array(value) for key, value in result.items()}
        return np.array(result)

    return wrapper


_func = numpify(jax.jit(loss_base))
_prop = numpify(jax.jit(partial(loss_base, do_props=True)))
_grad = numpify(jax.jit(jax.grad(loss_base)))
_hess = numpify(jax.jit(jax.hessian(loss_base)))
