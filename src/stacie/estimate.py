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
from .cutoff import CutoffCriterion, underfitting_criterion
from .model import ExpTailModel, SpectrumModel, guess
from .rpi import build_xgrid_exp, rpi_opt
from .spectrum import Spectrum

__all__ = ("FCutWarning", "Result", "estimate_acint", "fit_model_spectrum")


@attrs.define
class Result:
    """Container class holding all the results of the autocorrelation integral estimate."""

    spectrum: Spectrum = attrs.field()
    """The input spectrum from which the autocorrelation integral is estimated."""

    nfit: int = attrs.field()
    """
    The number of low-frequency spectrum data points used to fit the model.
    This value is the best among a series of tested cutoffs,
    i.e. by minimizing the cutoff criterion.
    """

    history: dict[int, dict[str]] = attrs.field()
    """History of nfit optimization.

    The key is ``nfit``, the number of frequencies fitted to.
    Each value is a dictionary returned by :func:`fit_model_spectrum`.
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
    *,
    fcutmax: float | None = None,
    maxscan: int = 100,
    nfitmin: int = 10,
    nfitmax_hard: int = 1000,
    model: SpectrumModel | None = None,
    cutoff_criterion: CutoffCriterion = underfitting_criterion,
    rng: np.random.Generator | None = None,
    nonlinear_budget: int = 10,
    verbose: bool = False,
) -> Result:
    """Estimate the integral of the autocorrelation function.

    It is recommended to leave the keyword arguments to their default values,
    except for methodological testing.

    This function fits a model to the low-frequency portion of the spectrum
    and derives an estimate of the autocorrelation (and its uncertainty) from the fit.
    The model is fitted to a part of the spectrum op to a cutoff frequency.
    Multiple cutoffs are tested, each resulting in another number of spectrum amplitudes in the fit,
    and the one that minimizes the ``cutoff_criterion`` is selected as the best solution.

    Parameters
    ----------
    spectrum
        The power spectrum and related metadata,
        used as inputs for the estimation of the autocorrelation integral.
        This object can be prepared with the function: :py:func:`stacie.spectrum.compute_spectrum`.
    fcutmax
        The maximum cutoff on the frequency axis (units of frequency),
        which corresponds to the largest value for ``nfit``.
    maxscan
        The maximum number of cutoffs to test.
        If 1, then only the given ``fcutmax`` is used.
    nfitmin
        The minimal amount of frequency data points to use in the fit.
    nfitmax_hard
        The maximal amount of frequency data points to use in the fit.
        This puts an upper bound on the computational cost of the fit.
        If this upper limit is stricter than that of ``fcutmax``, a warning is raised.
    model
        The model used to fit the low-frequency part of the spectrum.
        The default is an instance of :py:class:`stacie.model.ExpTailModel`.
    cutoff_criterion
        The criterion function that is minimized to find the best cutoff
        (and thus number of points included in the fit).
    rng
        A random number generator for sampling gueses of the nonlinear parameters.
        When not given, ``np.random.default_rng(42)`` is used.
        The seed is fixed by default for reproducibility.
    nonlinear_budget
        The number of samples to use for the nonlinear parameters is
        ``nonlinear_budget ** num_nonlinear``.
    verbose
        Set this to ``True`` to print progress information of the frequency cutoff search
        to the standard output.

    Returns
    -------
    result
        The inputs, intermediate results and outputs or the algorithm.
    """
    if model is None:
        model = ExpTailModel()
    if rng is None:
        rng = np.random.default_rng(42)
    history = {}

    if verbose and maxscan > 1:
        print("CUTOFF FREQUENCY SEARCH")
        print("   nfit   criterion  incumbent")
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
        evals = props["cost_hess_evals"]
        history[nfit] = props
        criterion = (
            props["criterion"] if (np.isfinite(evals).all() and (evals > 0).all()) else np.inf
        )
        if verbose and maxscan > 1:
            lowest_criterion = scratch.get("lowest_criterion")
            best = lowest_criterion is None or criterion < lowest_criterion
            if best:
                scratch["lowest_criterion"] = criterion
            print(f"{nfit:7d}  {criterion:10.1f}  {'<---' if best else ''}")
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
        nfits = build_xgrid_exp([nfitmin, nfitmax], maxscan)
        rpi_opt(compute_criterion, [0, len(nfits) - 1], mode="min")
        if any(np.isfinite(props["criterion"]) for props in history.values()):
            nfit = min((record["criterion"], key) for key, record in history.items())[1]
            if nfit == nfits[0]:
                warnings.warn(
                    "The lowest possible cutoff was selected. "
                    "This indicates that the time series are too short, "
                    "in which case the result is most likely biased.",
                    FCutWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "Could not find a suitable frequency cutoff. "
                "The resuts for the smallest cutoff are selected.",
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
) -> dict[str, NDArray]:
    """Optimize the parameter of a model for a given spectrum.

    The parameters are the attributes of the :py:class:`stacie.cost.LowFreqCost` class,
    except for the ones documented below.

    Parameters
    ----------
    nfit
        The number of low-frequency data points to use in the fit.
    cutoff_criterion
        The criterion function that is minimized to find the best cutoff
        (and thus number of points included in the fit).
    rng
        A random number generator for sampling gueses of the nonlinear parameters.
    nonlinear_budget
        The number of samples to use for the nonlinear parameters is
        ``nonlinear_budget ** num_nonlinear``

    Returns
    -------
    props
        A dictionary with various intermediate results of the cost function calculation,
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
    - ``cost_hess_evals``: Hessian eigenvalues
    - ``cost_hess_evecs``: Hessian eigenvectors
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

    The ExpTail model has the following additional properties:

    - ``corrtime``: estimate of the slowest time scale in the sequences
    - ``corrtime_var``: variance of the estimate of the slowest time scale
    - ``corrtime_std``: standard error of the estimate of the slowest time scale
    - ``exptail_simulation_time``: recommended simulation time based on the Exptail model
    - ``exptail_block_time``: recommended block time based on the Exptail model
    """
    # Maximize likelihood
    par_scales = model.get_par_scales(timestep, freqs[:nfit], amplitudes[:nfit])
    pars_init = guess(
        model,
        timestep,
        freqs[:nfit],
        amplitudes[:nfit],
        ndofs[:nfit],
        par_scales,
        rng,
        nonlinear_budget,
    )
    if not model.valid(pars_init):
        raise AssertionError("Infeasible guess")
    cost = LowFreqCost(timestep, freqs[:nfit], amplitudes[:nfit], ndofs[:nfit], model)
    conditioned_cost = ConditionedCost(cost, par_scales, 1.0)
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
    evals, evecs = np.linalg.eigh(props["cost_hess"])
    props["cost_hess_evals"] = evals
    props["cost_hess_evecs"] = evecs
    if (evals > 0).all() and np.isfinite(evals).all():
        half = evecs / np.sqrt(evals)
        props["covar"] = np.dot(half, half.T)
        props["pars_sensitivity"] = -np.dot(
            evecs, np.dot(evecs.T, props["cost_grad_sensitivity"]) / evals.reshape(-1, 1)
        )
    else:
        props["covar"] = np.full_like(props["cost_hess"], np.inf)
        props["pars_sensitivity"] = np.full((len(pars_opt), nfit), np.inf)

    # Derive estimates from model parameters.
    props.update(model.derive_props(props["pars"], props["covar"], props["pars_sensitivity"]))

    # Give recommendations for the block size and simulation time.
    props["sensitivity_block_time"] = 0.1 / freqs[nfit - 1]
    sweights = props["acint_sensitivity"] ** 2
    sweights_sum = sweights.sum()
    if np.isfinite(sweights_sum) and sweights_sum > 0:
        freq_relevant = np.dot(sweights, freqs[:nfit]) / sweights_sum
        props["sensitivity_simulation_time"] = 10 / freq_relevant
    else:
        props["sensitivity_simulation_time"] = np.inf

    # Compute remaining properties and derive the cutoff criterion
    props["pars_init"] = pars_init
    props["freqs_rest"] = freqs[nfit:]
    props["amplitudes_rest"] = amplitudes[nfit:]
    props["kappas_rest"] = 0.5 * ndofs[nfit:]
    props["criterion"] = cutoff_criterion(props)

    # Remove some intermediate properties to reduce the size of the Result object.
    del props["freqs"]
    del props["kappas"]
    del props["amplitudes"]
    del props["freqs_rest"]
    del props["amplitudes_rest"]
    del props["kappas_rest"]

    return props
