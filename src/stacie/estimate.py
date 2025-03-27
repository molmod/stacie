# Stacie is a STable AutoCorrelation Integral Estimator.
# Copyright (C) 2024-2025 The contributors of the Stacie Python Package.
# See the CONTRIBUTORS.md file in the project root for a full list of contributors.
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

from .conditioning import ConditionedCost
from .cost import LowFreqCost
from .cutoff import CutoffCriterion, cv2l_criterion
from .model import SpectrumModel, guess
from .rpi import build_xgrid_exp, rpi_opt
from .spectrum import Spectrum
from .utils import PositiveDefiniteError, robust_dot, robust_posinv

__all__ = ("FCutWarning", "Result", "estimate_acint", "fit_model_spectrum")


@attrs.define
class Result:
    """Container class holding all the results of the autocorrelation integral estimate."""

    spectrum: Spectrum = attrs.field()
    """The input spectrum from which the autocorrelation integral is estimated."""

    nfit: int = attrs.field()
    """
    The number of low-frequency spectrum data points used to fit the model.
    This value represents the optimal cutoff, selected from a range of tested cutoffs,
    by minimizing the given cutoff criterion.
    """

    history: dict[int, dict[str]] = attrs.field()
    """History of nfit optimization.

    The key is ``nfit``, which represents the number of frequencies fitted.
    Each value is a dictionary returned by :func:`fit_model_spectrum`,
    containing the intermediate results of the fitting process.
    """

    @property
    def props(self) -> dict[str]:
        """Properties computed from the fit up to the selected spectrum cutoff.

        This is a shortcut for ``history[nfit]``.
        See return value of :func:`fit_model_spectrum` for more details.
        """
        return self.history[self.nfit]

    @property
    def acint(self) -> float:
        """The autocorrelation integral."""
        return self.props["acint"]

    @property
    def acint_std(self) -> float:
        """The uncertainty of the autocorrelation integral."""
        return self.props["acint_std"]

    @property
    def corrtime_int(self) -> float:
        """The integrated correlation time."""
        return self.props["acint"] / (2 * self.spectrum.prefactor * self.spectrum.variance)

    @property
    def corrtime_int_std(self) -> float:
        """The uncertainty of the integrated correlation time."""
        return self.props["acint_std"] / (2 * self.spectrum.prefactor * self.spectrum.variance)


class FCutWarning(Warning):
    """Raised when there is an issue with the frequency cutoff."""


def estimate_acint(
    spectrum: Spectrum,
    model: SpectrumModel,
    *,
    fcutmax: float | None = None,
    maxscan: int = 100,
    nfitmin: int | None = None,
    nfitmax_hard: int = 1000,
    cutoff_criterion: CutoffCriterion = cv2l_criterion,
    rng: np.random.Generator | None = None,
    nonlinear_budget: int = 10,
    verbose: bool = False,
) -> Result:
    """Estimate the integral of the autocorrelation function.

    It is recommended to leave the keyword arguments to their default values,
    except for methodological testing.

    This function fits a model to the low-frequency portion of the spectrum and
    derives an estimate of the autocorrelation (and its uncertainty) from the fit.
    The model is fitted to a part of the spectrum up to a cutoff frequency.
    Multiple cutoff frequencies are tested,
    each resulting in a different number of spectrum amplitudes in the fit.
    The one that minimizes the ``cutoff_criterion`` is selected as the optimal solution.

    Parameters
    ----------
    spectrum
        The power spectrum and related metadata,
        used as inputs for the estimation of the autocorrelation integral.
        This object can be prepared with the function: :py:func:`stacie.spectrum.compute_spectrum`.
    model
        The model used to fit the low-frequency part of the spectrum.
    fcutmax
        The maximum cutoff on the frequency axis (in frequency units),
        corresponding to the largest value for ``nfit``.
    maxscan
        The maximum number of cutoffs to test during the optimization.
        If set to 1, only the given ``fcutmax`` is used, with no extra cutoff testing.
        A nearly logarithmic grid of ``nfit`` integers is generated otherwise with size ``maxscan``
        and the minimization of the cutoff_criterion will only try ``nfit`` values from this grid.
    nfitmin
        The minimum number of frequency data points to include in the fit.
        If not provided, this is set to 10 times the number of model parameters as a default.
    nfitmax_hard
        The maximum number of frequency data points to include in the fit.
        This imposes an upper bound on the computational cost of the fitting process.
        If this upper limit is stricter than that of ``fcutmax``, a warning is raised.
    cutoff_criterion
        The criterion function that is minimized to find the best cutoff frequency and,
        consequently, the optimal number of points included in the fit.
    rng
        A random number generator for sampling guesses of the nonlinear parameters.
        If not provided, ``np.random.default_rng(42)`` is used.
        The seed is fixed by default for reproducibility.
    nonlinear_budget
        The number of samples used for the nonlinear parameters, calculated as
        ``nonlinear_budget ** num_nonlinear``.
    verbose
        Set this to ``True`` to print progress information of the frequency cutoff search
        to the standard output.

    Returns
    -------
    result
        The inputs, intermediate results and outputs or the algorithm.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if nfitmin is None:
        nfitmin = 10 * model.npar
    history = {}

    if verbose and maxscan > 1:
        print("CUTOFF FREQUENCY SEARCH")
        print("   nfit   criterion  incumbent")
        # The scratch dictionary is used to print the incumbent minimum.
        scratch = {}

    def compute_criterion(ifit: int):
        """Criterion to be minimized to find the best frequency cutoff.

        Parameters
        ----------
        ifit
            The index in the nfits list to get the right cutoff.
        include_failed
            When ``True``, add failed fits to the history.
        """
        nfit = nfits[ifit]
        props = fit_model_spectrum(
            spectrum.timestep,
            spectrum.freqs,
            spectrum.amplitudes,
            spectrum.ndofs,
            nfit,
            model,
            cutoff_criterion,
            rng,
            nonlinear_budget,
        )
        history[nfit] = props
        evals = props["cost_hess_rescaled_evals"]
        criterion = props["criterion"]
        if not (np.isinf(criterion) or (np.isfinite(evals).all() and (evals > 0).all())):
            # Make the criterion infinite if the Hessian is not positive definite,
            # and the criterion did not flag it yet.
            props["criterion"] = np.inf
            criterion = np.inf
            props["criterion_error"] = "Hessian of full fit is not positive definite."
        if verbose and maxscan > 1:
            lowest_criterion = scratch.get("lowest_criterion")
            best = lowest_criterion is None or criterion < lowest_criterion
            if best:
                scratch["lowest_criterion"] = criterion
            line = f"{nfit:7d}  {criterion:10.1f}"
            if best:
                line += "  <---"
            criterion_error = props.get("criterion_error")
            if criterion_error is not None:
                line += f"  ({criterion_error})"
            print(line)
        return criterion

    nfitmax = len(spectrum.freqs) if fcutmax is None else int(spectrum.freqs.searchsorted(fcutmax))
    if nfitmax > nfitmax_hard:
        nfitmax = nfitmax_hard
        warnings.warn(
            "The maximum frequency cutoff is lowered to constrain "
            f"the maximum number of data points in the fit to {nfitmax}.",
            FCutWarning,
            stacklevel=2,
        )
    if nfitmax < nfitmin:
        raise ValueError("Too few data points for fit.")

    if maxscan == 1:
        nfit = nfitmax
        nfits = [nfit]
        compute_criterion(0)
    else:
        # Only fit to an even number of points, so the grid can be splitted into two equal halves.
        # This is required for the cv2 and cv2l criteria.
        nfits = [2 * i for i in build_xgrid_exp([nfitmin // 2, nfitmax // 2], maxscan)]
        rpi_opt(compute_criterion, [0, len(nfits) - 1], mode="min")
        if verbose:
            print()
        candidates = [
            (record["criterion"], key)
            for key, record in history.items()
            if np.isfinite(record["criterion"])
        ]
        if len(candidates) > 0:
            nfit = min(candidates)[1]
            if nfit == nfits[0]:
                warnings.warn(
                    "The lowest possible cutoff was selected. "
                    "This indicates that the time series are too short. "
                    "In this case, the result is most likely biased.",
                    FCutWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "Could not find a suitable frequency cutoff. "
                "The results for the smallest cutoff are selected.",
                FCutWarning,
                stacklevel=2,
            )
            nfit = nfits[0]
    return Result(spectrum, nfit, dict(sorted(history.items())))


def fit_model_spectrum(
    timestep: float,
    freqs: NDArray[float],
    amplitudes: NDArray[float],
    ndofs: NDArray[int],
    nfit: int,
    model: SpectrumModel,
    cutoff_criterion: CutoffCriterion,
    rng: np.random.Generator,
    nonlinear_budget: int,
) -> dict[str, NDArray | float]:
    """Optimize the parameter of a model for a given spectrum.

    The parameters are the attributes of the :py:class:`stacie.cost.LowFreqCost` class,
    except for the ones documented below.

    Parameters
    ----------
    nfit
        The number of low-frequency data points to include in the fit.
    cutoff_criterion
        The criterion function that is minimized to find the optimal cutoff
        (and thus determine the number of points to include in the fit).
    rng
        A random number generator for sampling guesses of the nonlinear parameters.
    nonlinear_budget
        The number of samples to use for the nonlinear parameters is
        ``nonlinear_budget ** num_nonlinear``

    Returns
    -------
    props
        A dictionary containing various intermediate results of the cost function calculation,
        computed for the optimized parameters.
        See Notes for details.

    Notes
    -----
    The returned dictionary contains the following items:

    - ``acint``: estimate of the autocorrelation integral
    - ``acint_var``: variance of the estimate of the autocorrelation integral
    - ``acint_std``: standard error of the estimate of the autocorrelation integral
    - ``cost_value``: cost function value
    - ``cost_grad``: cost Gradient vector (if ``deriv>=1``)
    - ``cost_grad_sensitivity``: sensitivity of the cost gradient to the amplitudes
    - ``cost_hess``: cost Hessian matrix (if ``deriv==2``)
    - ``cost_hess_scales``: Hessian rescaling vector, see ``robust_posinv``.
    - ``cost_hess_rescaled_evals``: Rescaled Hessian eigenvalues
    - ``cost_hess_rescaled_evecs``: Rescaled hessian eigenvectors
    - ``covar``: covariance matrix of the parameters
    - ``criterion``: value of the criterion whose minimizer determines the frequency cutoff
    - ``ll``: log likelihood
    - ``model``: model name used for the fit
    - ``pars_init``: initial guess of the parameters
    - ``pars``: optimized parameters
    - ``pars_sensitivity``: sensitivity of the parameters to the amplitudes.
    - ``sensitivity_simulation_time``: recommended simulation time based on the sensitivity analysis
    - ``sensitivity_block_time``: recommended block time based on the sensitivity analysis
    - ``thetas``: scale parameters for the gamma distribution
    - ``cutoff_criterion``: name of the cutoff criterion used to determine the frequency cutoff

    The ExpTail model has the following additional properties:

    - ``corrtime``: estimate of the slowest time scale in the sequences
    - ``corrtime_var``: variance of the estimate of the slowest time scale
    - ``corrtime_std``: standard error of the estimate of the slowest time scale
    - ``exptail_simulation_time``: recommended simulation time based on the Exptail model
    - ``exptail_block_time``: recommended block time based on the Exptail model
    """
    # Maximize likelihood
    model.configure_scales(timestep, freqs[:nfit], amplitudes[:nfit])
    pars_init = guess(
        model,
        timestep,
        freqs[:nfit],
        ndofs[:nfit],
        amplitudes[:nfit],
        model.par_scales,
        rng,
        nonlinear_budget,
    )
    if not model.valid(pars_init):
        raise AssertionError(f"Infeasible guess: {pars_init = }")
    cost = LowFreqCost(timestep, freqs[:nfit], ndofs[:nfit], amplitudes[:nfit], model)
    conditioned_cost = ConditionedCost(cost, model.par_scales, 1.0)
    opt = minimize(
        conditioned_cost.funcgrad,
        conditioned_cost.to_reduced(pars_init),
        jac=True,
        hess=conditioned_cost.hess,
        bounds=model.bounds(),
        method="trust-constr",
        options={"xtol": 1e-10, "gtol": 1e-10},
    )
    pars_opt = conditioned_cost.from_reduced(opt.x)
    props = cost.props(pars_opt, 2)

    # Compute the Hessian and its properties.
    try:
        hess_scales, evals, evecs, covar = robust_posinv(props["cost_hess"])
        par_sensitivity = -robust_dot(
            1 / hess_scales, 1 / evals, evecs, props["cost_grad_sensitivity"]
        )
    except PositiveDefiniteError:
        npar = len(pars_opt)
        hess_scales = np.full(npar, np.inf)
        evals = np.full(npar, np.inf)
        evecs = np.full((npar, npar), np.inf)
        covar = np.full((npar, npar), np.inf)
        par_sensitivity = np.full(npar, np.inf)
    props["cost_hess_scales"] = hess_scales
    props["cost_hess_rescaled_evals"] = evals
    props["cost_hess_rescaled_evecs"] = evecs
    props["covar"] = covar
    props["pars_sensitivity"] = par_sensitivity

    # Repeat the optimization for the first and the second half of the spectrum
    # if the covariance is positive definite.
    if np.isfinite(props["covar"]).all() and cutoff_criterion.half_opt:
        for label, ifirst, ilast in [("half1", 0, nfit // 2), ("half2", nfit // 2, nfit)]:
            cost = LowFreqCost(
                timestep, freqs[ifirst:ilast], ndofs[ifirst:ilast], amplitudes[ifirst:ilast], model
            )
            conditioned_cost = ConditionedCost(cost, model.par_scales, 1.0)
            opt = minimize(
                conditioned_cost.funcgrad,
                conditioned_cost.to_reduced(pars_opt),
                jac=True,
                hess=conditioned_cost.hess,
                bounds=model.bounds(),
                method="trust-constr",
                options={"xtol": 1e-10, "gtol": 1e-10},
            )
            pars_opt_half = conditioned_cost.from_reduced(opt.x)
            props[f"pars_{label}"] = pars_opt_half
            props[f"cost_hess_{label}"] = cost(pars_opt_half, 2)[2]

    # Derive estimates from model parameters.
    props.update(model.derive_props(props["pars"], props["covar"], props["pars_sensitivity"]))

    # Compute remaining properties and derive the cutoff criterion
    props["pars_init"] = pars_init
    props["freqs_rest"] = freqs[nfit:]
    props["amplitudes_rest"] = amplitudes[nfit:]
    props["kappas_rest"] = 0.5 * ndofs[nfit:]
    props["cutoff_criterion"] = cutoff_criterion.__name__
    props.update(cutoff_criterion(props))

    # Remove some intermediate properties to reduce the size of the Result object.
    del props["freqs"]
    del props["kappas"]
    del props["amplitudes"]
    del props["freqs_rest"]
    del props["amplitudes_rest"]
    del props["kappas_rest"]

    return props
