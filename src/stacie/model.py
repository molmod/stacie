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

import numpy as np
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
    def npar(cls):
        """The number of parameters."""
        return len(cls.bounds())

    @classmethod
    def valid(cls, pars) -> bool:
        """Returns True when the parameters are within the feasible region."""
        return all(pmin < par < pmax for (pmin, pmax), par in zip(cls.bounds(), pars, strict=True))

    @staticmethod
    def guess(freqs: NDArray[float], amplitudes: NDArray[float]) -> NDArray[float]:
        """Guess initial values of the parameters."""
        raise NotImplementedError

    @staticmethod
    def __call__(
        omegas: NDArray[float], pars: NDArray[float], deriv: int = 0
    ) -> list[NDArray[float]]:
        """The exponential tail spectrum model.

        Parameters
        ----------
        omegas
            This is ``2 * pi * freqs``, where ``freqs`` is the array with dimensionless frequencies,
            as obtained with ``numpy.fft.rfftfreq`` (maximum value 0.5),
            of which possibly a subset is taken.
        pars
            The parameters.
        deriv
            The order of derivatives to compute.

        Returns
        -------
        results
            A results list, index corresponds to order of derivative.
        """
        raise NotImplementedError

    @classmethod
    def derive_props(
        cls, pars: NDArray[float], covar: NDArray[float], timestep: float
    ) -> dict[str, NDArray[float]]:
        """Return additional properties derived from model-specific parameters."""
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
    def __call__(
        omegas: NDArray[float], pars: NDArray[float], deriv: int = 0
    ) -> list[NDArray[float]]:
        """See SpectrumModel.__call__"""
        if not isinstance(deriv, int):
            raise TypeError("Argument deriv must be integer.")
        if deriv < 0:
            raise ValueError("Argument deriv must be zero or positive.")
        if omegas.ndim != 1:
            raise TypeError("Argument omegas must be a 1D array.")
        acfint_short, acfint_tail, ctt = pars
        r = np.exp(-2 / ctt)
        cs = np.cos(omegas)
        denom = r**2 - 2 * r * cs + 1
        tail_model = ((1 - r) / (1 + r)) * (2 * (1 - r * cs) / denom - 1)
        results = [acfint_short + acfint_tail * tail_model]
        if deriv >= 1:
            tail_model_diff_r = -2 * (cs - 1) * (r**2 - 1) / denom**2
            r_diff_ctt = 2 * r / ctt**2
            tail_model_diff_ctt = tail_model_diff_r * r_diff_ctt
            results.append(
                np.array([np.ones(len(omegas)), tail_model, acfint_tail * tail_model_diff_ctt])
            )
        if deriv >= 2:
            tail_model_diff_r_r = 4 * (cs - 1) * (r**3 - 3 * r + 2 * cs) / denom**3
            r_diff_ctt_ctt = 4 * (1 - ctt) * r / ctt**4
            tail_model_diff_ctt_ctt = (
                tail_model_diff_r_r * r_diff_ctt**2 + tail_model_diff_r * r_diff_ctt_ctt
            )
            results.append(
                np.array(
                    [
                        [np.zeros(len(omegas)), np.zeros(len(omegas)), np.zeros(len(omegas))],
                        [np.zeros(len(omegas)), np.zeros(len(omegas)), tail_model_diff_ctt],
                        [
                            np.zeros(len(omegas)),
                            tail_model_diff_ctt,
                            acfint_tail * tail_model_diff_ctt_ctt,
                        ],
                    ]
                )
            )
        if deriv >= 3:
            raise ValueError("Third or higher derivatives are not supported.")
        return results

    @classmethod
    def derive_props(
        cls, pars: NDArray[float], covar: NDArray[float], timestep: float
    ) -> dict[str, NDArray[float]]:
        """Return additional properties derived from model-specific parameters."""
        acfint_var = covar[:2, :2].sum()
        corrtime_tail_var = covar[2, 2] * timestep**2
        return {
            "model": cls.name,
            "acfint": pars[:2].sum(),
            "acfint_var": acfint_var,
            "acfint_std": np.sqrt(acfint_var) if acfint_var >= 0 else np.inf,
            "corrtime_tail": pars[2] * timestep,
            "corrtime_tail_var": corrtime_tail_var,
            "corrtime_tail_std": (np.sqrt(corrtime_tail_var) if corrtime_tail_var >= 0 else np.inf),
        }


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
    def __call__(
        omegas: NDArray[float], pars: NDArray[float], deriv: int = 0
    ) -> list[NDArray[float]]:
        """See SpectrumModel.__call__"""
        if not isinstance(deriv, int):
            raise TypeError("Argument deriv must be integer.")
        if deriv < 0:
            raise ValueError("Argument deriv must be zero or positive.")
        if omegas.ndim != 1:
            raise TypeError("Argument omegas must be a 1D array.")
        npt = len(omegas)
        results = [np.full(npt, pars[0])]
        if deriv >= 1:
            results.append(np.ones((1, npt)))
        if deriv >= 2:
            results.append(np.zeros((1, 1, npt)))
        if deriv >= 3:
            raise ValueError("Third or higher derivatives are not supported.")
        return results

    @classmethod
    def derive_props(
        cls, pars: NDArray[float], covar: NDArray[float], timestep: float
    ) -> dict[str, NDArray[float]]:
        """Return additional properties derived from model-specific parameters."""
        return {
            "model": cls.name,
            "acfint": pars[0],
            "acfint_var": covar[0, 0],
            "acfint_std": np.sqrt(covar[0, 0]) if covar[0, 0] >= 0 else np.inf,
            "corrtime_tail": np.inf,
            "corrtime_tail_var": np.inf,
            "corrtime_tail_std": np.inf,
        }
