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

from .cost import LowFreqCost
from .cutobj import CutObj, cutobj_symcu
from .model import ExpTailModel, SpectrumModel
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
    """History of ncut optimization.

    Each value is a dictionary returned by ``fit_model_spectrum``.
    """

    @property
    def props(self) -> dict[str]:
        """Properties computed from the fit up to the selected spectrum cutoff.

        This is a shortcut for ``history[ncut]``.
        See return value of ``fit_model_spectrum`` for more details.
        """
        return self.history[self.ncut]


class HessianError(ValueError):
    """Raised when the Hessian is infinite or not strictly positive definite."""


def estimate_acfint(
    spectrum: Spectrum,
    *,
    fcut: float | None = None,
    maxscan: int = 100,
    ncutmin: int = 10,
    model: SpectrumModel | None = None,
    cutobj: CutObj = cutobj_symcu,
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
    model
        The model used to fit the low-frequency regime.
    cutobj
        Thu objective function used to determine the frequency cutoff.

    Returns
    -------
    result
        A ``Result`` instance with inputs, intermediate results and outputs.
    """
    if model is None:
        model = ExpTailModel()
    history = {}

    def objective(n: int):
        """Objective to be minimized to find the best frequency cutoff."""
        ncut = ncutmin + n * stride
        props = fit_model_spectrum(
            spectrum.timestep,
            spectrum.freqs[:ncut],
            spectrum.amplitudes[:ncut],
            spectrum.ndofs[:ncut],
            model,
            cutobj,
        )
        history[ncut] = props
        evals = props["hess_evals"]
        return props["obj"] if (np.isfinite(evals).all() and (evals > 0).all()) else np.inf

    ncutmax = len(spectrum.freqs) if fcut is None else spectrum.freqs.searchsorted(fcut)
    if ncutmax < ncutmin:
        raise ValueError("Too few data points for fit.")
    if maxscan == 1:
        ncut = ncutmax
        stride = 1
        objective(ncutmax - ncutmin)
    else:
        stride = max((ncutmax - ncutmin) // maxscan, 1)
        rpi_opt(objective, [0, maxscan], mode="min")
        if len(history) == 0:
            raise AssertionError("Could not find a suitable solution")
        ncut = min((record["obj"], key) for key, record in history.items())[1]

    return Result(spectrum, ncut, dict(sorted(history.items())))


def fit_model_spectrum(
    timestep: float,
    freqs: NDArray[float],
    amplitudes: NDArray[float],
    ndofs: NDArray[int],
    model: SpectrumModel,
    cutobj: CutObj,
) -> dict[str, NDArray]:
    """Optimize the parameter of a model for a given spectrum.

    The parameters are the attributes of the ``LowFeqCost`` class.

    Returns
    -------
    A dictionary with various intermediate results of the cost function calculation,
    computed for the optimized parameters.
    In addition to the properties returned by ``cost_low``,
    also the following are included:

    - ``hess``: the Hessian matrix at the solution.
    - ``hess_evals``: the Hessian eigenvalues.
    - ``hess_evecs``: the Hessian eigenvectors.
    - ``covar``: the covariance matrix of the parameters.
    - ``acfint``: the estimate of the ACF integral.
    - ``acfint_var``: the variance of estimate of the ACF integral.
    - ``acfint_std``: the standard error of estimate of the ACF integral.
    - ``corrtime_tail``: the slowest time scale in the sequences.
    - ``corrtime_tail_var``: the variance of estimate of the slowest time scale.
    - ``corrtime_tail_std``: the standard error of estimate of the slowest time scale.
    """
    # Maximize likelihood
    pars_init = model.guess(freqs, amplitudes)
    if not model.valid(pars_init):
        raise AssertionError("Infeasible guess")
    cost = LowFreqCost(timestep, freqs, amplitudes, ndofs, model, cutobj)
    opt = minimize(
        cost.func,
        pars_init,
        jac=cost.grad,
        hess=cost.hess,
        bounds=model.bounds(),
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
    model.update_props(props)
    return props
