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
from .cutoff import CutoffCriterion, underfitting_criterion
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
    cutoff_criterion: CutoffCriterion = underfitting_criterion,
    verbose: bool = False,
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
    cutoff_criterion
        The selected frequency cutoff minimizes this criterion.
    verbose
        Set this to ``True`` to print progress information of the frequency cutoff search
        to the standard output.

    Returns
    -------
    result
        A ``Result`` instance with inputs, intermediate results and outputs.
    """
    if model is None:
        model = ExpTailModel()
    history = {}

    if verbose and maxscan > 1:
        print("   ncut   criterion  incumbent")
        scratch = {}

    def compute_criterion(icut: int):
        """Criterion to be minimized to find the best frequency cutoff.

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
            spectrum.freqs,
            spectrum.amplitudes,
            spectrum.ndofs,
            ncut,
            model,
            cutoff_criterion,
        )
        evals = props["cost_hess_evals"]
        history[ncut] = props
        criterion = (
            props["criterion"] if (np.isfinite(evals).all() and (evals > 0).all()) else np.inf
        )
        if verbose and maxscan > 1:
            lowest_criterion = scratch.get("lowest_criterion")
            best = lowest_criterion is None or criterion < lowest_criterion
            if best:
                scratch["lowest_criterion"] = criterion
            print(f"{ncut:7d}  {criterion:10.1f}  {'<---' if best else ''}")
        return criterion

    ncutmax = len(spectrum.freqs) if fcutmax is None else int(spectrum.freqs.searchsorted(fcutmax))
    if ncutmax > ncutmax_hard:
        ncutmax = ncutmax_hard
        warnings.warn(
            "The maximum frequency cutoff is lowered to constrain "
            f"the maximum number of data points in the fit to {ncutmax}.",
            FCutWarning,
            stacklevel=2,
        )
    if ncutmax < ncutmin:
        raise ValueError("Too few data points for fit.")

    if maxscan == 1:
        ncut = ncutmax
        ncuts = [ncut]
        compute_criterion(0)
    else:
        ncuts = build_xgrid_exp([ncutmin, ncutmax], maxscan)
        rpi_opt(compute_criterion, [0, len(ncuts) - 1], mode="min")
        if any(np.isfinite(props["criterion"]) for props in history.values()):
            ncut = min((record["criterion"], key) for key, record in history.items())[1]
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
    ncut: int,
    model: SpectrumModel,
    cutoff_criterion: CutoffCriterion,
) -> dict[str, NDArray]:
    """Optimize the parameter of a model for a given spectrum.

    The parameters are the attributes of the ``LowFeqCost`` class,
    except for the ones documented below.

    Parameters
    ----------
    ncut
        The number of low-frequency data points to use in the fit.
    cutoff_criterion
        The selected frequency cutoff minimizes this criterion.

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

    - ``criterion``: the criterion whose minimizer determines the frequency cutoff.
    - ``cost_hess_evals``: the Hessian eigenvalues.
    - ``cost_hess_evecs``: the Hessian eigenvectors.
    - ``covar``: the covariance matrix of the parameters.
    - ``acfint``: the estimate of the ACF integral.
    - ``acfint_var``: the variance of estimate of the ACF integral.
    - ``acfint_std``: the standard error of estimate of the ACF integral.
    - ``corrtime_tail``: the slowest time scale in the sequences.
    - ``corrtime_tail_var``: the variance of estimate of the slowest time scale.
    - ``corrtime_tail_std``: the standard error of estimate of the slowest time scale.
    """
    # Maximize likelihood
    pars_init = model.guess(freqs[:ncut], amplitudes[:ncut])
    if not model.valid(pars_init):
        raise AssertionError("Infeasible guess")
    cost = LowFreqCost(timestep, freqs[:ncut], amplitudes[:ncut], ndofs[:ncut], model)
    opt = minimize(
        cost.funcgrad,
        pars_init,
        jac=True,
        hess=cost.hess,
        bounds=model.bounds(),
        method="trust-constr",
        options={"xtol": 1e-10, "gtol": 1e-10},
    )

    # Compute all properties and derive the cutoff criterion
    props = cost.props(opt.x, 2)
    props["freqs_rest"] = freqs[ncut:]
    props["amplitudes_rest"] = amplitudes[ncut:]
    props["ndofs_rest"] = ndofs[ncut:]
    props["criterion"] = cutoff_criterion(props)

    # Compute the Hessian and its properties.
    evals, evecs = np.linalg.eigh(props["cost_hess"])
    props["cost_hess_evals"] = evals
    props["cost_hess_evecs"] = evecs
    props["covar"] = np.linalg.inv(props["cost_hess"])

    # Derive estimates from model parameters.
    props.update(model.derive_props(props["pars"], props["covar"], props["timestep"]))
    return props
