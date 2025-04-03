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
from .cutoff import CutoffCriterion, CV2LCriterion, integral_to_cutoff, switch_func
from .model import SpectrumModel, guess
from .rpi import rpi_opt
from .spectrum import Spectrum
from .utils import PositiveDefiniteError, mixture_stats, robust_posinv

__all__ = ("FCutWarning", "Result", "estimate_acint", "fit_model_spectrum")


@attrs.define
class Result:
    """Container class holding all the results of the autocorrelation integral estimate."""

    spectrum: Spectrum = attrs.field()
    """The input spectrum from which the autocorrelation integral is estimated."""

    model: SpectrumModel = attrs.field()
    """The model used to fit the low-frequency part of the spectrum."""

    cutoff_criterion: CutoffCriterion = attrs.field()
    """The criterion used to select or weight cutoff frequencies."""

    props: dict[str] = attrs.field()
    """The properties of the selected cutoff frequency or ensemble of cutoff frequencies.

    Properties of this class derive their results from information in this dictionary.
    """

    history: list[dict[str]] = attrs.field()
    """History of the cutoff optimization.

    Each item is a dictionary returned by :func:`fit_model_spectrum`,
    containing the intermediate results of the fitting process.
    They are sorted from low to high cutoff frequency.
    """

    @property
    def ncut(self) -> int:
        """The number of points where the fitting weight is larger than 1/1000."""
        return self.props["ncut"]

    @property
    def fcut(self) -> int:
        """The (ensemble average) of the cutoff frequency."""
        return self.props["fcut"]

    @property
    def neff(self) -> int:
        """The effective number of frequencies used in the fit."""
        return self.props["weights"].sum()

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
    nfit_min: int | None = None,
    fcut_max: float | None = None,
    fcut_spacing: float = 0.5,
    switch_exponent: float = 20.0,
    cutoff_criterion: CutoffCriterion | None = None,
    rng: np.random.Generator | None = None,
    fcut_budget: int = 500,
    nonlinear_budget: int = 10,
    criterion_high: float = 100,
    verbose: bool = False,
) -> Result:
    r"""Estimate the integral of the autocorrelation function.

    It is recommended to leave the keyword arguments to their default values,
    except for methodological testing.

    This function fits a model to the low-frequency portion of the spectrum and
    derives an estimate of the autocorrelation (and its uncertainty) from the fit.
    It repeats this for a range of cutoff frequencies on a logarithmic grid.
    Finally, an ensemble average over all cutoffs is computed,
    by using ``-np.log`` of the cutoff criterion as weight.

    The cutoff frequency grid is logarithmically spaced,
    with the ratio between two successive cutoff frequencies given by

    .. math::

        \frac{f_{i+1}}{f_{i}} = \exp(s / p)

    where :math:`s` is ``fcut_spacing`` and :math:`p` is ``switch_exponent``.

    Parameters
    ----------
    spectrum
        The power spectrum and related metadata,
        used as inputs for the estimation of the autocorrelation integral.
        This object can be prepared with the function: :py:func:`stacie.spectrum.compute_spectrum`.
    model
        The model used to fit the low-frequency part of the spectrum.
    nfit_min
        The minimum effective number of frequency data points to include in the fit.
        (The effective number of points is the sum of weights in the smooth cutoff.)
        If not provided, this is set to 10 times the number of model parameters as a default.
    fcut_max
        If given, cutoffs beyond this maximum are not considered.
    fcut_spacing
        Dimensionless parameter that controls the spacing between cutoffs in the grid.
    switch_exponent
        Controls the sharpness of the cutoff.
        Lower values lead to a smoother cutoff, and require fewer cutoff grid points.
        Higher values sharpen the cutoff, reveal more details, but a finer cutoff grid.
    cutoff_criterion
        The criterion function that is minimized to find the best cutoff frequency and,
        consequently, the optimal number of points included in the fit.
        If not given, the default is an instance of :py:class:`stacie.cutoff.CV2LCriterion`.
    rng
        A random number generator for sampling guesses of the nonlinear parameters.
        If not provided, ``np.random.default_rng(42)`` is used.
        The seed is fixed by default for reproducibility.
    fcut_budget
        The maximum number of times a model is fitted to the spectrum with a different cutoff.
        When this maximum is reached, a warning is raised and the loop is interrupted.
    nonlinear_budget
        The number of samples used for the nonlinear parameters, calculated as
        ``nonlinear_budget ** num_nonlinear``.
    criterion_high
        An high increase in the cutoff criterion value,
        used to terminate the search for the cutoff frequency.
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
    if nfit_min is None:
        nfit_min = 10 * model.npar
    if cutoff_criterion is None:
        cutoff_criterion = CV2LCriterion()
    if spectrum.ndofs.max() < 16:
        warnings.warn(
            f"The number of degrees of freedom ({spectrum.ndofs.max()}) is too small. "
            "The results are most likely biased. "
            "Averaging spectra over at least 8 to increase the degrees of freedom.",
            FCutWarning,
            stacklevel=2,
        )

    def log(props):
        neff = props["neff"]
        criterion = props["criterion"]
        line = f"{neff:9.1f}   {criterion:10.1f}"
        msg = props.get("msg")
        if msg is not None:
            line += f"  ({msg})"
        print(line)

    deltaf = spectrum.freqs[1] - spectrum.freqs[0]
    fcut_min = integral_to_cutoff(nfit_min, switch_exponent) * deltaf - spectrum.freqs[0]
    fcut_ratio = np.exp(fcut_spacing / switch_exponent)
    history = []
    best_criterion = None
    if verbose:
        print("CUTOFF FREQUENCY SCAN")
        print("     neff    criterion")
    for icut in range(fcut_budget):
        fcut = fcut_min * fcut_ratio**icut
        if fcut_max is not None and fcut > fcut_max:
            break
        if fcut > spectrum.freqs[-1]:
            break
        # Compute the criterion for the current cutoff frequency.
        props = fit_model_spectrum(
            spectrum,
            model,
            fcut,
            switch_exponent,
            cutoff_criterion,
            rng,
            nonlinear_budget,
        )
        fcut_budget -= 1
        if verbose:
            log(props)
        if np.isfinite(props["criterion"]):
            history.append(props)
            criterion = props["criterion"]
            if best_criterion is None or criterion < best_criterion:
                best_criterion = criterion
            elif criterion > best_criterion + criterion_high:
                break
    else:
        warnings.warn("The maximum number of fits was exceeded.", RuntimeWarning, stacklevel=2)
    if verbose:
        print()

    if len(history) == 0:
        raise ValueError("The cutoff criterion could not be computed for any cutoff frequency.")

    # Weights and cutoff frequency
    criteria = np.array([props["criterion"] for props in history])
    criteria -= criteria.min()
    fcut_weights = np.exp(-criteria)
    fcut_weights /= fcut_weights.sum()
    for fcut_weight, props in zip(fcut_weights, history, strict=False):
        props["fcut_weight"] = fcut_weight
    weights = sum(
        fcut_weight * switch_func(spectrum.freqs, props["fcut"], switch_exponent)
        for fcut_weight, props in zip(fcut_weights, history, strict=True)
    )
    ncut = np.sum(weights > 1e-3)
    freqs = spectrum.freqs[:ncut]
    weights = weights[:ncut]

    # Parameters and covariance
    all_pars = np.array([props["pars"] for props in history])
    all_covars = np.array([props["covar"] for props in history])
    pars, covar = mixture_stats(all_pars, all_covars, fcut_weights)
    acint_lico = model.acint_lico
    acint_var = np.dot(acint_lico, np.dot(covar, acint_lico))

    props = {
        "fcut": np.dot(fcut_weights, [props["fcut"] for props in history]),
        "ncut": ncut,
        "weights": weights,
        "pars": pars,
        "covar": covar,
        "acint": np.dot(pars, acint_lico),
        "acint_var": acint_var,
        "acint_std": np.sqrt(acint_var),
        "amplitudes_model": model.compute(freqs, pars, deriv=1),
    }
    props.update(model.derive_props(props["pars"], props["covar"]))

    return Result(spectrum, model, cutoff_criterion, props, history)


def estimate_acint_opt(
    spectrum: Spectrum,
    model: SpectrumModel,
    *,
    fcut_max: float | None = None,
    maxscan: int = 100,
    nfit_min: int | None = None,
    nfit_max_hard: int = 1000,
    switch_exponent: float = 20.0,
    cutoff_criterion: CutoffCriterion | None = None,
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
    fcut_max
        The maximum cutoff on the frequency axis (in frequency units).
        If not given, then ``fcut_max`` is derived from ``nfit_max_hard``,
        and reduced to ``freq[-1]`` if that frequency is smaller.
        A warning is raised if there are more than ``nfit_max_hard`` frequency grid points.
    maxscan
        The maximum number of cutoffs to test during the optimization.
        (In practice, far fewer cutoffs need to be tested.)
        If set to 1, only the given ``fcut_max`` is used, with no extra cutoff testing.
        A logarithmic grid of ``nfit`` integers is generated otherwise with size ``maxscan``
        and the minimization of the cutoff_criterion will only try ``nfit`` values from this grid.
    nfit_min
        The minimum effective number of frequency data points to include in the fit.
        (The effective number of points is the sum of weights in the smooth cutoff.)
        If not provided, this is set to 10 times the number of model parameters as a default.
    nfit_max_hard
        The maximum effective number of frequency data points to include in the fit.
        This imposes an upper bound on the computational cost of the fitting process.
        If this upper limit is stricter than that of ``fcut_max``, a warning is raised.
    switch_exponent
        Controls the sharpness of the cutoff.
        Lower values lead to a smoother cutoff, and less noisy cutoff criterion.
        Higher values sharpen the cutoff but may introduce more local minima in the criterion.
    cutoff_criterion
        The criterion function that is minimized to find the best cutoff frequency and,
        consequently, the optimal number of points included in the fit.
        If not given, the default is an instance of :py:class:`stacie.cutoff.CV2LCriterion`.
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
    if nfit_min is None:
        nfit_min = 10 * model.npar
    if cutoff_criterion is None:
        cutoff_criterion = CV2LCriterion()
    if spectrum.ndofs.max() < 16:
        warnings.warn(
            f"The number of degrees of freedom ({spectrum.ndofs.max()}) is too small. "
            "The results are most likely biased. "
            "Averaging spectra over at least 8 to increase the degrees of freedom.",
            FCutWarning,
            stacklevel=2,
        )

    history = {}

    if verbose and maxscan > 1:
        print("CUTOFF FREQUENCY SEARCH")
        print("     neff    criterion  incumbent")
        # The scratch dictionary is used to print the incumbent minimum.
        scratch = {}

    def compute_criterion(icut: int):
        """Criterion to be minimized to find the best frequency cutoff.

        Parameters
        ----------
        ifit
            The index in the fcuts list to get the right cutoff.
        """
        fcut = fcuts[icut]
        props = fit_model_spectrum(
            spectrum,
            model,
            fcut,
            switch_exponent,
            cutoff_criterion,
            rng,
            nonlinear_budget,
        )
        criterion = props["criterion"]
        if np.isfinite(criterion):
            history[icut] = props
        # We set the fcut_weight to zero and only the props dictionary of the optimized cutoff
        # will receive a value of one in the end.
        props["fcut_weight"] = 0.0
        # Screen output and return
        if verbose and maxscan > 1:
            lowest_criterion = scratch.get("lowest_criterion")
            best = lowest_criterion is None or criterion < lowest_criterion
            if best:
                scratch["lowest_criterion"] = criterion
            neff = props["neff"]
            line = f"{neff:9.1f}   {criterion:10.1f}"
            if best:
                line += "  <---"
            msg = props.get("msg")
            if msg is not None:
                line += f"  ({msg})"
            print(line)
        return criterion

    # Determine minimum and maximum cutoff frequencies.
    deltaf = spectrum.freqs[1] - spectrum.freqs[0]
    fcut_min = integral_to_cutoff(nfit_min, switch_exponent) * deltaf - spectrum.freqs[0]
    fcut_max_hard = integral_to_cutoff(nfit_max_hard, switch_exponent) * deltaf - spectrum.freqs[0]
    if fcut_max is None:
        fcut_max = spectrum.freqs[-1]
    if fcut_max > fcut_max_hard:
        fcut_max = fcut_max_hard
        warnings.warn(
            "The maximum frequency cutoff is lowered to constrain "
            f"the maximum number of data points in the fit to {nfit_max_hard}.",
            FCutWarning,
            stacklevel=2,
        )
    if fcut_max < fcut_min:
        raise ValueError("The maximum cutoff frequency is lower than the minimal one.")
    if maxscan == 1:
        icut = 0
        fcuts = np.array([fcut_max])
        compute_criterion(0)
    else:
        fcuts = np.geomspace(fcut_min, fcut_max, maxscan)
        rpi_opt(compute_criterion, [0, len(fcuts) - 1], mode="min")
        if verbose:
            print()
        candidates = [
            (record["criterion"], key)
            for key, record in history.items()
            if np.isfinite(record["criterion"])
        ]
        if len(candidates) > 0:
            icut = min(candidates)[1]
            if icut == 0:
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
            icut = 0

    # Prepare the final result
    if len(history) == 0:
        raise ValueError("The cutoff criterion could not be computed for any cutoff frequency.")
    props = history[icut]
    props["fcut_weight"] = 1.0
    props["weights"] = switch_func(spectrum.freqs[: props["ncut"]], props["fcut"], switch_exponent)
    props["amplitudes_model"] = model.compute(
        spectrum.freqs[: props["ncut"]], props["pars"], deriv=1
    )
    return Result(
        spectrum,
        model,
        cutoff_criterion,
        props,
        [props for _, props in sorted(history.items())],
    )


def fit_model_spectrum(
    spectrum: Spectrum,
    model: SpectrumModel,
    fcut: float,
    switch_exponent: float,
    cutoff_criterion: CutoffCriterion,
    rng: np.random.Generator,
    nonlinear_budget: int,
) -> dict[str, NDArray | float]:
    """Optimize the parameter of a model for a given spectrum.

    Parameters
    ----------
    spectrum
        The spectrum object containing the input data.
    model
        The model to be fitted to the spectrum.
    fcut
        The cutoff frequency (in frequency units) used to construct the weights.
    switch_exponent
        Controls the sharpness of the cutoff.
        Lower values lead to a smoother cutoff.
        Higher values sharpen the cutoff.
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
    The returned dictionary contains the following items if the fit succeeds:

    - ``acint``: estimate of the autocorrelation integral
    - ``acint_var``: variance of the estimate of the autocorrelation integral
    - ``acint_std``: standard error of the estimate of the autocorrelation integral
    - ``cost_value``: cost function value
    - ``cost_grad``: cost Gradient vector (if ``deriv>=1``)
    - ``cost_hess``: cost Hessian matrix (if ``deriv==2``)
    - ``cost_hess_scales``: Hessian rescaling vector, see ``robust_posinv``.
    - ``cost_hess_rescaled_evals``: Rescaled Hessian eigenvalues
    - ``cost_hess_rescaled_evecs``: Rescaled hessian eigenvectors
    - ``covar``: covariance matrix of the parameters
    - ``switch_exponent``: exponent used to construct the cutoff
    - ``criterion``: value of the criterion whose minimizer determines the frequency cutoff
    - ``ll``: log likelihood
    - ``pars_init``: initial guess of the parameters
    - ``pars``: optimized parameters

    The ``ExpTailModel`` has the following additional properties:

    - ``corrtime_exp``: estimate of the slowest time scale in the sequences
    - ``corrtime_exp_var``: variance of the estimate of the slowest time scale
    - ``corrtime_exp_std``: standard error of the estimate of the slowest time scale
    - ``exptail_simulation_time``: recommended simulation time based on the Exptail model
    - ``exptail_block_time``: recommended block time based on the Exptail model

    If the fit fails, the following properties are set:

    - ``criterion``: infinity
    - ``msg``: error message
    """
    # Create a switching function for a smooth cutoff
    weights = switch_func(spectrum.freqs, fcut, switch_exponent)
    ncut = (weights >= 1e-3).nonzero()[0][-1]
    freqs = spectrum.freqs[:ncut]
    ndofs = spectrum.ndofs[:ncut]
    amplitudes = spectrum.amplitudes[:ncut]
    weights = weights[:ncut]

    # Construct the initial guess for the model parameters.
    model.configure_scales(spectrum.timestep, freqs, amplitudes)
    pars_init = guess(freqs, ndofs, amplitudes, weights, model, rng, nonlinear_budget)

    # Sanity check of the initial guess
    props = {
        "fcut": fcut,
        "ncut": ncut,
        "switch_exponent": switch_exponent,
        "neff": weights.sum(),
        "pars_init": pars_init,
    }
    cost = LowFreqCost(freqs, ndofs, amplitudes, weights, model)
    if not (model.valid(pars_init) and np.isfinite(cost(pars_init, 0)[0])):
        props["criterion"] = np.inf
        props["msg"] = "init: Invalid guess"
        return props

    # Optimize the parameters
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
    props["pars"] = pars_opt
    props.update(cost.props(pars_opt, 2))

    # Compute the Hessian and its properties.
    try:
        hess_scales, evals, evecs, covar = robust_posinv(props["cost_hess"])
    except PositiveDefiniteError as exc:
        props["criterion"] = np.inf
        props["msg"] = f"opt: Hessian {exc.args[0]}"
        return props
    props["cost_hess_scales"] = hess_scales
    props["cost_hess_rescaled_evals"] = evals
    props["cost_hess_rescaled_evecs"] = evecs
    props["covar"] = covar

    # Derive estimates from model parameters.
    acint_lico = model.acint_lico
    props["acint"] = np.dot(pars_opt, acint_lico)
    props["acint_var"] = np.dot(acint_lico, np.dot(covar, acint_lico))
    props["acint_std"] = np.sqrt(props["acint_var"])
    props.update(model.derive_props(props["pars"], props["covar"]))

    # Add remaining properties and derive the cutoff criterion
    props.update(cutoff_criterion(spectrum, model, props))

    # Done
    return props
