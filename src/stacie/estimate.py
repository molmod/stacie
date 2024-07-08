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
"""Algorithm to estimate the autocorrelation integral."""

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .model import LowFreqCost
from .rpi import rpi_opt
from .spectrum import Spectrum

__all__ = ("Result", "HessianError", "estimate_acfint", "fit_model_spectrum")


@attrs.define
class Result:
    """Container class holding all the results of the ACF integral estimate."""

    spectrum: Spectrum = attrs.field()
    """The input spectrum from which the ACF integral is estimated."""

    ncut: int = attrs.field()
    """The low-pass cutoff index of the frequency axis, to suppress periodic boundary artifacts."""

    history: dict[int, dict[str]] = attrs.field()
    """History of ncut optimization."""

    @property
    def props(self) -> dict[str]:
        """Properties computed from the fit up to the selected spectrum cutoff."""
        return self.history[self.ncut]

    @property
    def acfint(self) -> float:
        """Integral of the autocorrelation integral."""
        return self.props["pars"][:2].sum()

    @property
    def acfint_var(self) -> float:
        """Variance of the  of the autocorrelation integral."""
        return self.props["covar"][:2, :2].sum()

    @property
    def acfint_std(self) -> float:
        """Uncertainty of the autocorrelation integral."""
        return np.sqrt(self.acfint_var)

    @property
    def corrtime_tail(self) -> float:
        """Slowest time scale in the signal."""
        return self.props["pars"][2]

    @property
    def corrtime_tail_var(self) -> float:
        """Variance of the slowest time scale in the signal."""
        return self.props["covar"][2, 2]

    @property
    def corrtime_tail_std(self) -> float:
        """Uncertianty of the slowest time scale in the signal."""
        return np.sqrt(self.corrtime_tail_var)


class HessianError(ValueError):
    """Raised when the Hessian is infinite or not strictly positive definite."""


def estimate_acfint(
    spectrum: Spectrum,
    *,
    fcut: float | None = None,
    maxscan: int = 100,
    ncutmin: int = 10,
) -> Result:
    """Estimate the integral of the autocorrelation function.

    It is recommended to leave the keyword arguments to their default values,
    except for methodological testing.

    Parameters
    ----------
    spectrum
        A ``Spectrum`` instance holding all the inputs for the estimation of the ACF integral.
    fcut
        The highest value of the spectrum to consider
    maxscan
        The maximum number of cutoffs to test.
        If 1, then only the given fcut is used.
    ncutmin
        The minimal amount of frequency data points to use in the fit.

    Returns
    -------
    result
        A ``Result`` instance with inputs, intermediate results and outputs.
    """
    history = {}

    def objective(n: int, check_hess: bool = False):
        ncut = ncutmin + n * stride
        props = fit_model_spectrum(
            spectrum.timestep,
            spectrum.freqs[:ncut],
            spectrum.amplitudes[:ncut],
            spectrum.ndofs[:ncut],
        )
        if check_hess:
            # When scanning, only good minima are saved and treated as valid
            hess_evals = props["hess_evals"]
            if not (np.isfinite(hess_evals).all() and (hess_evals > 0).all()):
                return np.inf
        history[ncut] = props
        return props["obj"]

    ncutmax = len(spectrum.freqs) if fcut is None else spectrum.freqs.searchsorted(fcut)
    if ncutmax < ncutmin:
        raise ValueError("Too few data points for fit.")
    if maxscan == 1:
        ncut = ncutmax
        stride = 1
        if not np.isfinite(objective(ncutmax - ncutmin)):
            raise ValueError("Fit failed")
    else:
        stride = max((ncutmax - ncutmin) // maxscan, 1)
        rpi_opt(objective, [0, maxscan], mode="min")
        if len(history) == 0:
            raise AssertionError("Could not find a suitable solution")
        ncut = min((record["obj"], key) for key, record in history.items())[1]

    return Result(spectrum, ncut, dict(sorted(history.items())))


def fit_model_spectrum(
    timestep: float, freqs: NDArray[float], amplitudes: NDArray[float], ndofs: NDArray[int]
) -> dict[str, NDArray]:
    # Maximize likelihood
    cost = LowFreqCost(timestep, freqs, amplitudes, ndofs)
    pars_init = cost.guess
    if not cost.valid(pars_init):
        raise AssertionError("Infeasible guess")
    opt = minimize(
        cost.func,
        pars_init,
        jac=cost.grad,
        hess=cost.hess,
        bounds=cost.bounds,
        method="trust-constr",
    )

    # Compute all properties
    hess = cost.hess(opt.x)
    evals, evecs = np.linalg.eigh(hess)
    props = cost.prop(opt.x)
    props["hess"] = hess
    props["hess_evals"] = evals
    props["hess_evecs"] = evecs
    props["covar"] = np.linalg.inv(hess)
    return props
