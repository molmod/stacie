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

__all__ = ("SpectrumModel", "ExpTailModel", "WhiteNoiseModel")


@attrs.define
class SpectrumModel:
    """Abstract base class for spectrum models.

    Subclasses must override the attribute ``name``
    and the methods ``bounds``, ``guess``, ``compute`` and ``derive_props``.
    """

    # Attributes used for pre-conditioning the parameter space.
    timestep: float = attrs.field(default=1.0)
    amplitude_scale: float = attrs.field(default=1.0)

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

    def precondition(
        self,
        timestep: float,
        amplitudes: NDArray[float],
    ) -> NDArray[float]:
        """Precondition the parameter space and the cost function value."""
        self.timestep = timestep
        self.amplitude_scale = np.median(abs(amplitudes[amplitudes != 0]))
        if self.amplitude_scale == 0:
            self.amplitude_scale = 1

    def guess(
        self,
        freqs: NDArray[float],
        amplitudes: NDArray[float],
        ndofs: NDArray[float],
    ) -> NDArray[float]:
        """Guess initial values of the parameters."""
        raise NotImplementedError

    def compute(
        self, omegas: NDArray[float], pars: NDArray[float], deriv: int = 0
    ) -> list[NDArray[float]]:
        """Compute the amplitudes of the spectrum model.

        Parameters
        ----------
        omegas
            This is ``2 * pi * freqs * timestep``,
            where ``freqs * timestep`` is the array with dimensionless frequencies,
            as obtained with ``numpy.fft.rfftfreq`` (maximum value 0.5),
            of which possibly a subset is taken.
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
        self, pars: NDArray[float], covar: NDArray[float]
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
    - :math:`\tau_\text{exp} / h`:
      The correlation time of the exponential tail
      {cite:p}`sokal_1997_monte`.
      This is in dimensionless time units, as if the time step is 1.
    """

    @property
    def name(self):
        return "exptail"

    @staticmethod
    def bounds() -> list[tuple[float, float]]:
        """Return parameter bounds for the optimizer."""
        return [(0, np.inf), (0, np.inf), (0, np.inf)]

    def guess(
        self,
        freqs: NDArray[float],
        amplitudes: NDArray[float],
        ndofs: NDArray[float],
    ) -> NDArray[float]:
        """Guess initial values of the parameters."""
        # This is implemented as a 1D non-linear optimization of corrtime.
        # For each tau, the two remaining parameters are found with weighted linear regression.
        omegas = 2 * np.pi * self.timestep * freqs
        amplitudes = amplitudes / self.amplitude_scale
        amplitudes_std = amplitudes / np.sqrt(0.5 * ndofs)

        def linear_fit_rmsd(
            corrtime_tau: float, return_pars: bool = False
        ) -> float | tuple[float, float]:
            pars = np.array([1.0 / self.amplitude_scale, 1.0 / self.amplitude_scale, corrtime_tau])
            basis = self.compute(omegas, pars, 1)[1][:2] / self.amplitude_scale
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
        ctmin = 0.1 / omegas[-1]
        ctmax = 1 / omegas[1]
        nscan = 10
        cts = np.exp(np.linspace(np.log(ctmin), np.log(ctmax), nscan))
        rmsds = np.array([linear_fit_rmsd(ct) for ct in cts])

        # Try to identify and refine a bracket
        ibest = rmsds.argmin()
        if ibest not in (0, nscan - 1):
            bracket = cts[ibest - 1 : ibest + 2]

            # Refine the bracket
            minres = minimize_scalar(
                linear_fit_rmsd, bracket, method="golden", options={"xtol": 0.1 * ctmin}
            )
            if minres.success:
                return linear_fit_rmsd(minres.x, True)

        # Minimize_scalar did not work: take the best from the scan.
        return linear_fit_rmsd(cts[ibest], True)

    def compute(
        self, omegas: NDArray[float], pars: NDArray[float], deriv: int = 0
    ) -> list[NDArray[float]]:
        """See :py:meth:`SpectrumModel.compute`."""
        if not isinstance(deriv, int):
            raise TypeError("Argument deriv must be integer.")
        if deriv < 0:
            raise ValueError("Argument deriv must be zero or positive.")
        if omegas.ndim != 1:
            raise TypeError("Argument omegas must be a 1D array.")

        # Unpack parameters and undo preconditioning
        acint_short, acint_tail, ct = pars
        acint_short *= self.amplitude_scale
        acint_tail *= self.amplitude_scale

        r = np.exp(-1 / ct)
        cs = np.cos(omegas)
        denom = r**2 - 2 * r * cs + 1
        tail_model = (1 - r) ** 2 / denom
        results = [acint_short + acint_tail * tail_model]
        if deriv >= 1:
            tail_model_diff_r = -2 * (cs - 1) * (r**2 - 1) / denom**2
            r_diff_ct = r / ct**2
            tail_model_diff_ct = tail_model_diff_r * r_diff_ct
            results.append(
                np.array(
                    [
                        np.ones(len(omegas)) * self.amplitude_scale,
                        tail_model * self.amplitude_scale,
                        acint_tail * tail_model_diff_ct,
                    ]
                )
            )
        if deriv >= 2:
            tail_model_diff_r_r = 4 * (cs - 1) * (r**3 - 3 * r + 2 * cs) / denom**3
            r_diff_ct_ct = (1 - 2 * ct) * r / ct**4
            tail_model_diff_ct_ct = (
                tail_model_diff_r_r * r_diff_ct**2 + tail_model_diff_r * r_diff_ct_ct
            )
            results.append(
                np.array(
                    [
                        [np.zeros(len(omegas)), np.zeros(len(omegas)), np.zeros(len(omegas))],
                        [
                            np.zeros(len(omegas)),
                            np.zeros(len(omegas)),
                            tail_model_diff_ct * self.amplitude_scale,
                        ],
                        [
                            np.zeros(len(omegas)),
                            tail_model_diff_ct * self.amplitude_scale,
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
        acint = pars[:2].sum() * self.amplitude_scale
        acint_var = covar[:2, :2].sum() * self.amplitude_scale**2
        corrtime = pars[2] * self.timestep
        corrtime_var = covar[2, 2] * self.timestep**2
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

    def guess(
        self,
        freqs: NDArray[float],
        amplitudes: NDArray[float],
        ndofs: NDArray[float],
    ) -> NDArray[float]:
        """Guess initial values of the parameters."""
        return np.array([amplitudes.mean()]) / self.amplitude_scale

    def compute(
        self, omegas: NDArray[float], pars: NDArray[float], deriv: int = 0
    ) -> list[NDArray[float]]:
        """See :py:meth:`SpectrumModel.compute`."""
        if not isinstance(deriv, int):
            raise TypeError("Argument deriv must be integer.")
        if deriv < 0:
            raise ValueError("Argument deriv must be zero or positive.")
        if omegas.ndim != 1:
            raise TypeError("Argument omegas must be a 1D array.")
        npt = len(omegas)
        results = [np.full(npt, pars[0] * self.amplitude_scale)]
        if deriv >= 1:
            results.append(np.ones((1, npt)) * self.amplitude_scale)
        if deriv >= 2:
            results.append(np.zeros((1, 1, npt)))
        if deriv >= 3:
            raise ValueError("Third or higher derivatives are not supported.")
        return results

    def derive_props(
        self, pars: NDArray[float], covar: NDArray[float]
    ) -> dict[str, NDArray[float]]:
        """Return additional properties derived from model-specific parameters."""
        return {
            "model": self.name,
            "acint": pars[0] * self.amplitude_scale,
            "acint_var": covar[0, 0] * self.amplitude_scale**2,
            "acint_std": np.sqrt(covar[0, 0]) * self.amplitude_scale
            if covar[0, 0] >= 0
            else np.inf,
            "corrtime": np.inf,
            "corrtime_var": np.inf,
            "corrtime_std": np.inf,
        }
