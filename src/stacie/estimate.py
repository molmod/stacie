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

import warnings

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .cost import LowFreqCost
from .cutobj import CutObj, cutobj_symcu
from .model import ExpTailModel, SpectrumModel
from .rpi import build_xgrid_exp, rpi_opt
from .spectrum import Spectrum

__all__ = ("Result", "FCutWarning", "estimate_acfint", "fit_model_spectrum")


@attrs.define
class Result:
    """Container class holding all the results of the ACF integral estimate."""

    spectrum: Spectrum = attrs.field()
    """The input spectrum from which the ACF integral is estimated."""

    ncut: int = attrs.field()
    """The low-pass cutoff index of the frequency axis, to suppress periodic boundary artifacts."""

    history: dict[int, dict[str]] = attrs.field()
    """History of ncut optimization.

    Each value is a dictionary returned by :func:`fit_model_spectrum`.
    """

    @property
    def props(self) -> dict[str]:
        """Properties computed from the fit up to the selected spectrum cutoff.

        This is a shortcut for ``history[ncut]``.
        See return value of :func:`fit_model_spectrum` for more details.
        """
        return self.history[self.ncut]


class FCutWarning(Warning):
    """Raised when there is an issue with the frequency cutoff.

    The algorithm will try to continue, but the results are unlikely to be useful.
    """


def estimate_acfint(
    spectrum: Spectrum,
    *,
    fcutmax: float | None = None,
    maxscan: int = 100,
    ncutmin: int = 10,
    ncutmax_hard: int = 1000,
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
    fcutmax
        The maximum cutoff on the frequency axis (unit of frequency).
    maxscan
        The maximum number of cutoffs to test.
        If 1, then only the given fcutmax is used.
    ncutmin
        The minimal amount of frequency data points to use in the fit.
    ncutmax_hard
        The maximal amount of frequency data points to use in the fit.
        This upper limit puts an upper bound on the computational cost of the fit.
        If this upper limit is stricter than that of fcutmax, a warning is raised.
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

    def objective(icut: int):
        """Objective to be minimized to find the best frequency cutoff.

        Parameters
        ----------
        icut
            The index in the ncuts list to get the right cutoff.
        include_failed
            When ``True``, add failed fits to the history.
        """
        ncut = ncuts[icut]
        props = fit_model_spectrum(
            spectrum.timestep,
            spectrum.freqs[:ncut],
            spectrum.amplitudes[:ncut],
            spectrum.ndofs[:ncut],
            model,
            cutobj,
        )
        evals = props["hess_evals"]
        history[ncut] = props
        result = props["obj"] if (np.isfinite(evals).all() and (evals > 0).all()) else np.inf
        print(ncut, result)
        return result

    ncutmax = len(spectrum.freqs) if fcutmax is None else int(spectrum.freqs.searchsorted(fcutmax))
    if ncutmax > ncutmax_hard:
        ncutmax = ncutmax_hard
        warnings.warn(
            "The maximum frequency cutoff is lowered to constrain "
            f"the maximum number of data points in the fit to {ncutmax}",
            FCutWarning,
            stacklevel=2,
        )
    if ncutmax < ncutmin:
        raise ValueError("Too few data points for fit.")

    if maxscan == 1:
        ncut = ncutmax
        ncuts = [ncut]
        objective(0)
    else:
        ncuts = build_xgrid_exp([ncutmin, ncutmax], maxscan)
        rpi_opt(objective, [0, len(ncuts) - 1], mode="min")
        if any(np.isfinite(props["obj"]) for props in history.values()):
            ncut = min((record["obj"], key) for key, record in history.items())[1]
        else:
            warnings.warn(
                "Could not find a suitable frequency cutoff. "
                "The resuts for the smallest cutoff are selected.",
                FCutWarning,
                stacklevel=2,
            )
            ncut = ncuts[0]

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
    props
        A dictionary with various intermediate results of the cost function calculation,
        computed for the optimized parameters.
        See Notes for details.

    Notes
    -----
    In addition to the properties returned by :func:`stacie.cost.cost_low`,
    the returned dictionary also contains the following items:

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
