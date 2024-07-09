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
"""Models to fit the low-frequency part of the spectrum."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike as JArrayLike
from numpy.typing import NDArray

__all__ = ("SpectrumModel", "ExpTailModel", "WhiteNoiseModel")


class SpectrumModel:
    """Abstract base class for spectrum models.

    Subclasses must override the attribute ``name``
    and the methods ``bounds``, ``guess``, ``__call__`` and ``update_props``.
    """

    name: str = NotImplemented

    @staticmethod
    def bounds() -> list[tuple[float, float]]:
        """Parameter bounds for the optimizer."""
        raise NotImplementedError

    @classmethod
    def valid(cls, pars) -> bool:
        """Returns True when the parameters are within the feasible region."""
        return all(pmin < par < pmax for (pmin, pmax), par in zip(cls.bounds(), pars, strict=True))

    @staticmethod
    def guess(freqs: NDArray[float], amplitudes: NDArray[float]) -> NDArray[float]:
        """Guess initial values of the parameters."""
        raise NotImplementedError

    @staticmethod
    def __call__(omegas: JArrayLike, pars: JArrayLike) -> jax.Array:
        """The exponential tail spectrum model.

        Parameters
        ----------
        omegas
            This is ``2 * pi * freqs``, where ``freqs`` is the array with dimensionless frequencies,
            as obtained with ``numpy.fft.rfftfreq`` (maximum value 0.5),
            of which possibly a subset is taken.
        pars
            The three positive model parameters: acfint_short, acfint_tail, corrtime_tail.

        Returns
        -------
        amplitudes_model
            The amplitudes of the model spectrum at the given omegas.
        """
        raise NotImplementedError

    @classmethod
    def update_props(cls, props: dict[str]):
        """Add results items to the props dictionary derived from the model parameters."""
        raise NotImplementedError


class ExpTailModel(SpectrumModel):
    """The exponential tail model for the spectrum.

    This model is derived under the following assumptions:

    - RFFT is used to compute the spectrum.
    - The autocorrelation consists of two contributions:
        - A quickly decaying part with no specific functional form,
          but confined within a small time lag domain centered at t=0.
        - An slow exponentially decaying part with a yet unknown characteristic time scale.
    """

    name: str = "exptail"

    @staticmethod
    def bounds() -> list[tuple[float, float]]:
        """Parameter bounds for the optimizer."""
        return [(0, np.inf), (0, np.inf), (0, np.inf)]

    @staticmethod
    def guess(freqs: NDArray[float], amplitudes: NDArray[float]) -> NDArray[float]:
        """Guess initial values of the parameters."""
        acfint_coarse = amplitudes[0]
        tau = 2 / freqs[-1]
        return np.array([acfint_coarse / 2, acfint_coarse / 2, tau])

    @staticmethod
    def __call__(omegas: JArrayLike, pars: JArrayLike) -> jax.Array:
        """See SpectrumModel.__call__"""
        acfint_short, acfint_tail, corrtime_tail = pars
        cosines = 2 * jnp.cos(omegas)
        ratio = jnp.exp(-2 / corrtime_tail)
        tail_model = (2 - ratio * cosines) / (1 + ratio**2 - ratio * cosines)
        # tail_model = 1 / ((2/corrtime_tail)**2 + omegas**2)
        tail_model /= tail_model[0]
        return acfint_tail * tail_model + acfint_short

    @classmethod
    def update_props(cls, props: dict[str]):
        """Add results items to the props dictionary derived from the model parameters."""
        props["model"] = cls.name
        props["acfint"] = props["pars"][:2].sum()
        acfint_var = props["covar"][:2, :2].sum()
        props["acfint_var"] = acfint_var
        props["acfint_std"] = np.sqrt(acfint_var) if acfint_var >= 0 else np.inf
        props["corrtime_tail"] = props["pars"][2]
        corrtime_tail_var = props["covar"][2, 2]
        props["corrtime_tail_var"] = corrtime_tail_var
        props["corrtime_tail_std"] = (
            np.sqrt(corrtime_tail_var) if corrtime_tail_var >= 0 else np.inf
        )


class WhiteNoiseModel(SpectrumModel):
    """One may fall back to this model if one suspects there are no time correlations."""

    name: str = "white"

    @staticmethod
    def bounds() -> list[tuple[float, float]]:
        """Parameter bounds for the optimizer."""
        return [(0, np.inf)]

    @staticmethod
    def guess(freqs: NDArray[float], amplitudes: NDArray[float]) -> NDArray[float]:
        """Guess initial values of the parameters."""
        return np.array([amplitudes.mean()])

    @staticmethod
    def __call__(omegas: JArrayLike, pars: JArrayLike) -> jax.Array:
        """See SpectrumModel.__call__"""
        return jnp.full_like(omegas, pars[0])

    @classmethod
    def update_props(cls, props: dict[str]):
        """Add results items to the props dictionary derived from the model parameters."""
        props["model"] = cls.name
        props["acfint"] = props["pars"][0].sum()
        acfint_var = props["covar"][0, 0].sum()
        props["acfint_var"] = acfint_var
        props["acfint_std"] = np.sqrt(acfint_var) if acfint_var >= 0 else np.inf
        props["corrtime_tail"] = np.inf
        props["corrtime_tail_var"] = np.inf
        props["corrtime_tail_std"] = np.inf
