# STACIE is a STable AutoCorrelation Integral Estimator.
# Copyright (C) 2024-2025 The contributors of the STACIE Python Package.
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

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, root_scalar

from .conditioning import ConditionedCost
from .cost import LowFreqCost
from .cutoff import WEIGHT_EPS, CutoffCriterion, CV2LCriterion, switch_func
from .model import SpectrumModel, guess
from .spectrum import Spectrum
from .utils import PositiveDefiniteError, UnitConfig, label_unit, mixture_stats, robust_posinv

__all__ = ("Result", "estimate_acint", "fit_model_spectrum", "summarize_results")


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
    """The properties marginalized over the ensemble of cutoff frequencies.

    Properties of this class derive their results from information in this dictionary.
    See docstring of :func:`fit_model_spectrum` for details.
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
        """The weighted average of the cutoff frequency."""
        return self.props["fcut"]

    @property
    def neff(self) -> int:
        """The weighted average of the effective number of frequencies used in the fit."""
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
        return self.props["acint"] / self.spectrum.variance

    @property
    def corrtime_int_std(self) -> float:
        """The uncertainty of the integrated correlation time."""
        return self.props["acint_std"] / self.spectrum.variance

    @property
    def corrtime_exp(self) -> float:
        """The exponential correlation time."""
        if "corrtime_exp" not in self.props:
            raise ValueError("The model does not provide an exponential correlation time.")
        return self.props["corrtime_exp"]

    @property
    def corrtime_exp_std(self) -> float:
        """The uncertainty of the exponential correlation time."""
        if "corrtime_exp_std" not in self.props:
            raise ValueError("The model does not provide an exponential correlation time.")
        return self.props["corrtime_exp_std"]


def estimate_acint(
    spectrum: Spectrum,
    model: SpectrumModel,
    *,
    neff_min: int | None = None,
    neff_max: int | None = 1000,
    fcut_min: float | None = None,
    fcut_max: float | None = None,
    fcut_spacing: float = 0.5,
    switch_exponent: float = 8.0,
    cutoff_criterion: CutoffCriterion | None = None,
    rng: np.random.Generator | None = None,
    nonlinear_budget: int = 100,
    criterion_high: float = 100,
    verbose: bool = False,
    uc: UnitConfig | None = None,
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

        \frac{f_{k+1}}{f_{k}} = \exp(g_\text{sp} / \beta)

    where :math:`g_\text{sp}` is ``fcut_spacing`` and :math:`\beta` is ``switch_exponent``.

    Parameters
    ----------
    spectrum
        The power spectrum and related metadata,
        used as inputs for the estimation of the autocorrelation integral.
        This object can be prepared with the function: :py:func:`stacie.spectrum.compute_spectrum`.
    model
        The model used to fit the low-frequency part of the spectrum.
    neff_min
        The minimum effective number of frequency data points to include in the fit.
        (The effective number of points is the sum of weights in the smooth cutoff.)
        If not provided, this is set to 5 times the number of model parameters as a default.
    neff_max
        The maximum number of effective points to include in the fit.
        This parameter limits the total computational cost.
        Set to None to disable this stopping criterion.
    fcut_min
        The minimum cutoff frequency to use.
        If given, this parameter can only increase the minimal cutoff derived from ``neff_min``.
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
        If not given, the default is an instance of
        :py:class:`stacie.cutoff.CV2LCriterion`.
    rng
        A random number generator for sampling guesses of the nonlinear parameters.
        If not provided, ``np.random.default_rng(42)`` is used.
        The seed is fixed by default for reproducibility.
    nonlinear_budget
        The number of samples used for the nonlinear parameters, calculated as
        ``nonlinear_budget ** num_nonlinear``.
    criterion_high
        An high increase in the cutoff criterion value,
        used to terminate the search for the cutoff frequency.
    verbose
        Set this to ``True`` to print progress information of the frequency cutoff search
        to the standard output.
    uc
        Unit configuration object used to format the screen output.
        If not provided, the default unit configuration is used.
        See :py:class:`stacie.utils.UnitConfig` for details.
        This only affects the screen output (if any) and not the results!

    Returns
    -------
    result
        The inputs, intermediate results and outputs or the algorithm.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if neff_min is None:
        neff_min = 5 * model.npar
    if cutoff_criterion is None:
        cutoff_criterion = CV2LCriterion()
    if uc is None:
        uc = UnitConfig()

    def log(props):
        neff = props["neff"]
        criterion = props["criterion"]
        line = f"{neff:9.1f}  {criterion:10.1f}    {fcut / uc.freq_unit:{uc.freq_fmt}}"
        msg = props.get("msg")
        if msg is not None:
            line += f"  ({msg})"
        print(line)

    # Determine range of cutoff frequencies
    fcut_min0 = root_scalar(
        (lambda f: switch_func(spectrum.freqs, f, switch_exponent).sum() - neff_min),
        bracket=(spectrum.freqs[1], spectrum.freqs[-1]),
    ).root
    fcut_min = fcut_min0 if fcut_min is None else max(fcut_min0, fcut_min)
    fcut_max0 = root_scalar(
        (lambda f: switch_func(spectrum.freqs[-1], f, switch_exponent) - WEIGHT_EPS),
        bracket=(spectrum.freqs[1], spectrum.freqs[-1]),
    ).root
    fcut_max = fcut_max0 if fcut_max is None else min(fcut_max0, fcut_max)

    # Iterate over the cutoff frequency_grid
    fcut_ratio = np.exp(fcut_spacing / switch_exponent)
    history = []
    best_criterion = None
    if verbose:
        print(f"CUTOFF FREQUENCY SCAN {cutoff_criterion.name}")
        print(f"     neff   criterion  {label_unit('fcut', uc.freq_unit_str):>10s}")
        print("---------  ----------  ----------")
    icut = 0
    while True:
        fcut = fcut_min * fcut_ratio**icut
        if fcut_max is not None and fcut > fcut_max:
            if verbose:
                print(f"Reached the maximum cutoff frequency ({fcut_max}).")
            break
        if fcut > spectrum.freqs[-1]:
            if verbose:
                print(f"Reached end of spectrum ({spectrum.freqs[-1]}).")
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
        if verbose:
            log(props)
        if np.isfinite(props["criterion"]):
            history.append(props)
            criterion = props["criterion"]
            if best_criterion is None or criterion < best_criterion:
                best_criterion = criterion
            elif criterion > best_criterion + criterion_high and len(history) > 10:
                if verbose:
                    print(
                        "Cutoff criterion exceeds incumbent + margin: "
                        f"{best_criterion:.1f} + {criterion_high:.1f}."
                    )
                break
        if neff_max is not None and props["neff"] > neff_max:
            if verbose:
                print(f"Reached the maximum number of effective points ({neff_max}).")
            break
        if props.get("stop", False):
            if verbose:
                print("Scan stopped by cutoff criterion.")
            break
        icut += 1

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
    ncut = np.sum(weights > WEIGHT_EPS)
    freqs = spectrum.freqs[:ncut]
    weights = weights[:ncut]

    props = {
        # Fixed parameters
        "ncut": ncut,
        "weights": weights,
        "switch_exponent": switch_exponent,
        # Some properties can only be mixed directly.
        "fcut": np.dot(fcut_weights, [props["fcut"] for props in history]),
        "cost_zscore": np.dot(fcut_weights, [props["cost_zscore"] for props in history]),
        "criterion_zscore": np.dot(fcut_weights, [props["criterion_zscore"] for props in history]),
    }
    # Use fcut_weights to mix model parameters and their covariance
    props["pars"], props["pars_covar"] = mixture_stats(
        np.array([props["pars"] for props in history]),
        np.array([props["pars_covar"] for props in history]),
        fcut_weights,
    )
    # Derive other properties from the mixed parameters, instead of mixing them directly.
    # This seems to be slightly better,
    # likely because the fcut_weights are based on parameter vectors only.
    props["amplitudes_model"] = model.compute(freqs, props["pars"], deriv=1)
    _finalize_props(props, model)

    result = Result(spectrum, model, cutoff_criterion, props, history)

    if verbose:
        print()
        print(summarize_results(result, uc=uc))
    return result


def fit_model_spectrum(
    spectrum: Spectrum,
    model: SpectrumModel,
    fcut: float,
    switch_exponent: float,
    cutoff_criterion: CutoffCriterion,
    rng: np.random.Generator,
    nonlinear_budget: int,
) -> dict[str, NDArray | float]:
    """Optimize the parameter of a model for a given spectrum and cutoff frequency.

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
    - ``cost_expected``: expected value of the cost function
    - ``cost_var``: expected variance of the cost function
    - ``cost_zscore``: z-score of the cost function
    - ``switch_exponent``: exponent used to construct the cutoff
    - ``criterion``: value of the criterion whose minimizer determines the frequency cutoff
    - ``criterion_expected``: expected value of the criterion
    - ``criterion_var``: expected variance of the criterion
    - ``criterion_zscore``: z-score of the criterion
    - ``ll``: log likelihood
    - ``pars_init``: initial guess of the parameters
    - ``pars``: optimized parameters
    - ``pars_covar``: covariance matrix of the parameters

    If the model can derive the exponential correlation time,
    The following properties are also included:

    - ``corrtime_exp``: exponential correlation time, the slowest time scale in the sequences
    - ``corrtime_exp_var``: variance of the estimated exponential correlation time
    - ``corrtime_exp_std``: standard error of the estimated exponential correlation time
    - ``exp_simulation_time``: recommended simulation time based on the exponential correlation time
    - ``exp_block_time``: recommended block time based on the exponential correlation time

    The ``ExpPolyModel`` has the following additional properties:

    - ``log_acint``: the logarithm of the autocorrelation integral
    - ``log_acint_var``: variance of the logarithm of the autocorrelation integral
    - ``log_acint_std``: standard error of the logarithm of the autocorrelation integral

    If the fit fails, the following properties are set:

    - ``criterion``: infinity
    - ``msg``: error message
    """
    # Create a switching function for a smooth cutoff
    weights = switch_func(spectrum.freqs, fcut, switch_exponent)
    ncut = (weights >= WEIGHT_EPS).sum()
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
    if not model.valid(pars_init):
        props["criterion"] = np.inf
        props["msg"] = "init: Invalid initial parameters"
        return props

    # Construct cost function and further validate initial guess
    cost = LowFreqCost(freqs, ndofs, amplitudes, weights, model)
    if not np.isfinite(cost(pars_init, deriv=0)[0]):
        props["criterion"] = np.inf
        props["msg"] = "init: Infinite cost for initial parameters"
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
    props["cost_value"], props["cost_grad"], props["cost_hess"] = cost(pars_opt, deriv=2)

    # Compute the Hessian and its properties.
    try:
        hess_scales, evals, evecs, pars_covar = robust_posinv(props["cost_hess"])
    except PositiveDefiniteError as exc:
        props["criterion"] = np.inf
        props["msg"] = f"opt: Hessian {exc.args[0]}"
        return props
    props["cost_hess_scales"] = hess_scales
    props["cost_hess_rescaled_evals"] = evals
    props["cost_hess_rescaled_evecs"] = evecs
    props["pars_covar"] = pars_covar

    # Compute the cutoff criterion
    props.update(cutoff_criterion(spectrum, model, props))

    # Compute the z-scores
    props["cost_expected"], props["cost_var"] = cost.expected(pars_opt)
    props["cost_zscore"] = (
        (props["cost_value"] - props["cost_expected"]) / np.sqrt(props["cost_var"])
        if np.isfinite(props["cost_value"])
        else np.inf
    )
    props["criterion_zscore"] = (
        (props["criterion"] - props["criterion_expected"]) / np.sqrt(props["criterion_var"])
        if np.isfinite(props["criterion"])
        else np.inf
    )

    _finalize_props(props, model)

    # Done
    return props


def _finalize_props(props: dict[str, NDArray | float], model: SpectrumModel):
    """Add remaining properties in-place."""
    props.update(model.derive_props(props["pars"], props["pars_covar"]))
    std_props = {}
    for key, value in props.items():
        if key.endswith("_var"):
            std_props[f"{key[:-4]}_std"] = np.sqrt(value) if value >= 0 else np.inf
    props.update(std_props)


def summarize_results(res: Result | list[Result], uc: UnitConfig | None = None):
    """Return a string summarizing the Result object."""
    if isinstance(res, Result):
        res = [res]
    if uc is None:
        uc = UnitConfig()
    texts = []
    for r in res:
        text = GENERAL_TEMPLATE.format(
            r=r,
            uc=uc,
            model=r.model.name,
            cutoff_criterion=r.cutoff_criterion.name,
            timestep=r.spectrum.timestep / uc.time_unit,
            simtime=r.spectrum.nstep * r.spectrum.timestep / uc.time_unit,
            acint=r.acint / uc.acint_unit,
            acint_std=r.acint_std / uc.acint_unit,
            corrtime_int=r.corrtime_int / uc.time_unit,
            corrtime_int_std=r.corrtime_int_std / uc.time_unit,
            npar=len(r.props["pars"]),
            maxdof=r.spectrum.ndofs.max(),
            fcut=r.fcut / uc.freq_unit,
            neff_threshold=20 * len(r.props["pars"]),
            cost_zscore=r.props["cost_zscore"],
            criterion_zscore=r.props["criterion_zscore"],
        )
        if "corrtime_exp" in r.props:
            text += EXPONENTIAL_TEMPLATE.format(
                uc=uc,
                corrtime_exp=r.props["corrtime_exp"] / uc.time_unit,
                corrtime_exp_std=r.props["corrtime_exp_std"] / uc.time_unit,
                exp_simulation_time=r.props["exp_simulation_time"] / uc.time_unit,
                exp_simulation_time_std=r.props["exp_simulation_time_std"] / uc.time_unit,
                exp_block_time=r.props["exp_block_time"] / uc.time_unit,
                exp_block_time_std=r.props["exp_block_time_std"] / uc.time_unit,
            )
        texts.append(text)
    return "\n---\n".join(texts)


GENERAL_TEMPLATE = """\
INPUT TIME SERIES
    Time step:                     {timestep:{uc.time_fmt}} {uc.time_unit_str}
    Simulation time:               {simtime:{uc.time_fmt}} {uc.time_unit_str}
    Maximum degrees of freedom:    {maxdof}

MAIN RESULTS
    Autocorrelation integral:      {acint:{uc.acint_fmt}} ± {acint_std:{uc.acint_fmt}} \
{uc.acint_unit_str}
    Integrated correlation time:   {corrtime_int:{uc.time_fmt}} ± {corrtime_int_std:{uc.time_fmt}} \
{uc.time_unit_str}

SANITY CHECKS (weighted averages over cutoff grid)
    Effective number of points:    {r.neff:.1f} (ideally > {neff_threshold:d})
    Regression cost Z-score:       {cost_zscore:.1f} (ideally < 2)
    Cutoff criterion Z-score:      {criterion_zscore:.1f} (ideally < 2)

MODEL {model} | CUTOFF CRITERION {cutoff_criterion}
    Number of parameters:          {npar}
    Average cutoff frequency:      {fcut:{uc.freq_fmt}} {uc.freq_unit_str}
"""

EXPONENTIAL_TEMPLATE = """\
    Exponential correlation time:  {corrtime_exp:{uc.time_fmt}} ± {corrtime_exp_std:{uc.time_fmt}} \
{uc.time_unit_str}

RECOMMENDED SIMULATION SETTINGS (EXPONENTIAL CORR. TIME)
    Block time:                    {exp_block_time:{uc.time_fmt}} ± \
{exp_block_time_std:{uc.time_fmt}} {uc.time_unit_str}
    Simulation time:               {exp_simulation_time:{uc.time_fmt}} ± \
{exp_simulation_time_std:{uc.time_fmt}} {uc.time_unit_str}
"""
