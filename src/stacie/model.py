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

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

__all__ = ("ExpTailModel", "SpectrumModel", "WhiteNoiseModel")


@attrs.define
class SpectrumModel:
    """Abstract base class for spectrum models.

    Subclasses must override the attribute ``name``
    and the methods ``bounds``, ``guess``, ``compute`` and ``derive_props``.
    """

    @property
    def name(self):
        raise NotImplementedError

    @staticmethod
    def bounds() -> list[tuple[float, float]]:
        """Return parameter bounds for the optimizer."""
        raise NotImplementedError

    @classmethod
    def npar(cls):
        """Return the number of parameters."""
        return len(cls.bounds())

    @classmethod
    def valid(cls, pars) -> bool:
        """Return ``True`` when the parameters are within the feasible region."""
        return all(pmin < par < pmax for (pmin, pmax), par in zip(cls.bounds(), pars, strict=True))

    def get_scales_low(
        self, timestep: float, freqs: NDArray[float], amplitudes: NDArray[float]
    ) -> dict[str, float]:
        """Helper for the get_par_scales method."""
        return {
            "time_scale": 1 / freqs[-1],
            "amp_scale": np.median(abs(amplitudes[amplitudes != 0])),
        }

    def get_par_scales(
        self, timestep: float, freqs: NDArray[float], amplitudes: NDArray[float]
    ) -> NDArray[float]:
        """Return the scales of the parameters and the cost function."""
        raise NotImplementedError

    def guess(
        self,
        freqs: NDArray[float],
        amplitudes: NDArray[float],
        ndofs: NDArray[float],
        timestep: float,
    ) -> NDArray[float]:
        """Guess initial values of the parameters."""
        raise NotImplementedError

    def compute(
        self, freqs: NDArray[float], timestep: float, pars: NDArray[float], deriv: int = 0
    ) -> list[NDArray[float]]:
        """Compute the amplitudes of the spectrum model.

        Parameters
        ----------
        freqs
            The frequencies for which the model spectrum amplitudes are computed.
        timestep
            The time step of the sequences used to compute the spectrum.
            It may be used to convert the frequency to the dimensionless
            normalized frequency :math:`2\\pi h f=2\\pi k/N`,
            where :math:`h` is the timestep,
            :math:`f` is the frequency,
            :math:`k` is the frequency index in the discrete Fourier transform,
            and :math:`N` is the number of samples in the input time series.
        pars
            The parameters.
        deriv
            The maximum order of derivatives to compute: 0, 1 or 2.

        Returns
        -------
        results
            A results list, index corresponds to order of derivative.
        """
        raise NotImplementedError

    def derive_props(
        self, pars: NDArray[float], covar: NDArray[float], timestep: float
    ) -> dict[str, NDArray[float]]:
        """Return additional properties derived from model-specific parameters."""
        raise NotImplementedError


@attrs.define
class ExpTailModel(SpectrumModel):
    r"""The Exponential Tail model for the spectrum.

    This model is derived under the following assumptions:

    - RFFT is used to compute the spectrum.
    - The autocorrelation consists of two contributions:
        - A quickly decaying part with no specific functional form,
          but confined within a small time lag domain centered at t=0.
        - An slow exponentially decaying part with a yet unknown characteristic time scale.

    The mathematic form is:

    .. math::

        C^\text{exp-tail}_k
            \approx
            a_\text{short} + a_\text{tail} \frac{1-r}{1+r} \left(
                2\frac{1 - r\cos(2\pi k/N)}{1 - 2r\cos(2\pi k/N) + r^2} - 1
            \right)

    with

    .. math::

        r = \exp\left(-\frac{h}{\tau_\text{exp}}\right)

    where :math:`h` is the time step and :math:`k` is the index of the RFFT vector.
    (Note that in the implementation :math:`2\pi k/N` is computed as :math:`2\pi h f`,
    where :math:`f` is the frequency.)

    The three parameters of the model in this implementation are:

    - :math:`a_\text{short}`:
      The prefactor for the short-term (white noise) component.
    - :math:`a_\text{tail}`:
      The prefactor for the exponential tail component.
    - :math:`\tau_\text{exp}`:
      The correlation time of the exponential tail
      {cite:p}`sokal_1997_monte`.
    """

    @property
    def name(self):
        return "exptail"

    @staticmethod
    def bounds() -> list[tuple[float, float]]:
        """Return parameter bounds for the optimizer."""
        return [(0, np.inf), (0, np.inf), (0, np.inf)]

    def get_par_scales(
        self, timestep: float, freqs: NDArray[float], amplitudes: NDArray[float]
    ) -> NDArray[float]:
        """Return the scales of the parameters and the cost function."""
        sl = self.get_scales_low(timestep, freqs, amplitudes)
        return np.array([sl["amp_scale"], sl["amp_scale"], sl["time_scale"]])

    def guess(
        self,
        freqs: NDArray[float],
        amplitudes: NDArray[float],
        ndofs: NDArray[float],
        timestep: float,
    ) -> NDArray[float]:
        """Guess initial values of the parameters."""
        # This is implemented as a 1D non-linear optimization of corrtime.
        # For each tau, the two remaining parameters are found with weighted linear regression.
        amplitudes_std = amplitudes / np.sqrt(0.5 * ndofs)

        def linear_fit_rmsd(
            corrtime_exp: float, return_pars: bool = False
        ) -> float | tuple[float, float]:
            pars = np.array([1.0, 1.0, corrtime_exp])
            basis = self.compute(freqs, timestep, pars, 1)[1][:2]
            pars[:2] = np.linalg.lstsq(
                (basis / amplitudes_std).T,
                amplitudes / amplitudes_std,
                # For compatibility with numpy < 2.0
                rcond=-1,
            )[0]
            if not self.valid(pars):
                pars[:2] = amplitudes[0] / 2
            if return_pars:
                return pars
            model = np.dot(pars[:2], basis)
            return (((model - amplitudes) / amplitudes_std) ** 2).mean()

        # Perform a quick scan of correlation times
        corrtime_min = 0.1 / freqs[-1]
        corrtime_max = 1 / freqs[1]
        nscan = 10
        corrtimes = np.exp(np.linspace(np.log(corrtime_min), np.log(corrtime_max), nscan))
        rmsds = np.array([linear_fit_rmsd(corrtime) for corrtime in corrtimes])

        # Try to identify and refine a bracket
        ibest = rmsds.argmin()
        if ibest not in (0, nscan - 1):
            bracket = corrtimes[ibest - 1 : ibest + 2]

            # Refine the bracket
            minres = minimize_scalar(
                linear_fit_rmsd, bracket, method="golden", options={"xtol": 0.1 * corrtime_min}
            )
            if minres.success:
                return linear_fit_rmsd(minres.x, True)

        # Minimize_scalar did not work: take the best from the scan.
        return linear_fit_rmsd(corrtimes[ibest], True)

    def compute(
        self, freqs: NDArray[float], timestep: float, pars: NDArray[float], deriv: int = 0
    ) -> list[NDArray[float]]:
        """See :py:meth:`SpectrumModel.compute`."""
        if not isinstance(deriv, int):
            raise TypeError("Argument deriv must be integer.")
        if deriv < 0:
            raise ValueError("Argument deriv must be zero or positive.")
        if freqs.ndim != 1:
            raise TypeError("Argument freqs must be a 1D array.")
        acint_short, acint_tail, corrtime = pars
        r = np.exp(-timestep / corrtime)
        cs = np.cos(2 * np.pi * timestep * freqs)
        denom = r**2 - 2 * r * cs + 1
        tail_model = (1 - r) ** 2 / denom
        results = [acint_short + acint_tail * tail_model]
        if deriv >= 1:
            tail_model_diff_r = -2 * (cs - 1) * (r**2 - 1) / denom**2
            r_diff_ct = r * timestep / corrtime**2
            tail_model_diff_ct = tail_model_diff_r * r_diff_ct
            results.append(
                np.array([np.ones(len(freqs)), tail_model, acint_tail * tail_model_diff_ct])
            )
        if deriv >= 2:
            tail_model_diff_r_r = 4 * (cs - 1) * (r**3 - 3 * r + 2 * cs) / denom**3
            r_diff_ct_ct = (1 - 2 * corrtime / timestep) * r * (timestep / corrtime**2) ** 2
            tail_model_diff_ct_ct = (
                tail_model_diff_r_r * r_diff_ct**2 + tail_model_diff_r * r_diff_ct_ct
            )
            results.append(
                np.array(
                    [
                        [np.zeros(len(freqs)), np.zeros(len(freqs)), np.zeros(len(freqs))],
                        [
                            np.zeros(len(freqs)),
                            np.zeros(len(freqs)),
                            tail_model_diff_ct,
                        ],
                        [
                            np.zeros(len(freqs)),
                            tail_model_diff_ct,
                            acint_tail * tail_model_diff_ct_ct,
                        ],
                    ]
                )
            )
        if deriv >= 3:
            raise ValueError("Third or higher derivatives are not supported.")
        return results

    def derive_props(
        self, pars: NDArray[float], covar: NDArray[float]
    ) -> dict[str, NDArray[float]]:
        """Return additional properties derived from model-specific parameters."""
        acint = pars[:2].sum()
        acint_var = covar[:2, :2].sum()
        corrtime = pars[2]
        corrtime_var = covar[2, 2]
        return {
            "model": self.name,
            "acint": acint,
            "acint_var": acint_var,
            "acint_std": np.sqrt(acint_var) if acint_var >= 0 else np.inf,
            "corrtime": corrtime,
            "corrtime_var": corrtime_var,
            "corrtime_std": (np.sqrt(corrtime_var) if corrtime_var >= 0 else np.inf),
        }


@attrs.define
class WhiteNoiseModel(SpectrumModel):
    """One may fall back to this model if one suspects there are no time correlations."""

    @property
    def name(self):
        return "white"

    @staticmethod
    def bounds() -> list[tuple[float, float]]:
        """Return parameter bounds for the optimizer."""
        return [(0, np.inf)]

    def get_par_scales(
        self, timestep: float, freqs: NDArray[float], amplitudes: NDArray[float]
    ) -> NDArray[float]:
        """Return the scales of the parameters and the cost function."""
        sl = self.get_scales_low(timestep, freqs, amplitudes)
        return np.array([sl["amp_scale"]])

    def guess(
        self,
        freqs: NDArray[float],
        amplitudes: NDArray[float],
        ndofs: NDArray[float],
        timestep: float,
    ) -> NDArray[float]:
        """Guess initial values of the parameters."""
        return np.array([amplitudes.mean()])

    def compute(
        self, freqs: NDArray[float], timestep: float, pars: NDArray[float], deriv: int = 0
    ) -> list[NDArray[float]]:
        """See :py:meth:`SpectrumModel.compute`."""
        if not isinstance(deriv, int):
            raise TypeError("Argument deriv must be integer.")
        if deriv < 0:
            raise ValueError("Argument deriv must be zero or positive.")
        if freqs.ndim != 1:
            raise TypeError("Argument freqs must be a 1D array.")
        results = [np.full(len(freqs), pars[0])]
        if deriv >= 1:
            results.append(np.ones((1, len(freqs))))
        if deriv >= 2:
            results.append(np.zeros((1, 1, len(freqs))))
        if deriv >= 3:
            raise ValueError("Third or higher derivatives are not supported.")
        return results

    def derive_props(
        self, pars: NDArray[float], covar: NDArray[float]
    ) -> dict[str, NDArray[float]]:
        """Return additional properties derived from model-specific parameters."""
        return {
            "model": self.name,
            "acint": pars[0],
            "acint_var": covar[0, 0],
            "acint_std": np.sqrt(covar[0, 0]) if covar[0, 0] >= 0 else np.inf,
            "corrtime": np.inf,
            "corrtime_var": np.inf,
            "corrtime_std": np.inf,
        }
