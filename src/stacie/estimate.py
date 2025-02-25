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

__all__ = ("Result", "FCutWarning", "estimate_acint", "fit_model_spectrum")


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
    def corrtime_exp(self) -> float:
        """The exponential correlation time."""
        return self.props["corrtime"]

    @property
    def corrtime_exp_std(self) -> float:
        """The uncertainty of the exponential correlation time."""
        return self.props["corrtime_std"]

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
    history = {}

    if verbose and maxscan > 1:
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

    Returns
    -------
    props
        A dictionary with various intermediate results of the cost function calculation,
        computed for the optimized parameters.
        See Notes for details.

    Notes
    -----
    The returned dictionary contains the following items:

    - ``acint``: the estimate of the autocorrelation integral.
    - ``acint_var``: the variance of the estimate of the autocorrelation integral.
    - ``acint_std``: the standard error of the estimate of the autocorrelation integral.
    - ``corrtime``: the estimate of the slowest time scale in the sequences.
    - ``corrtime_var``: the variance of the estimate of the slowest time scale.
    - ``corrtime_std``: the standard error of the estimate of the slowest time scale.
    - ``cost_value``: the cost function value.
    - ``cost_grad``: the cost Gradient vector (if ``deriv>=1``).
    - ``cost_hess``: the cost Hessian matrix (if ``deriv==2``).
    - ``cost_hess_evals``: the Hessian eigenvalues.
    - ``cost_hess_evecs``: the Hessian eigenvectors.
    - ``covar``: the covariance matrix of the parameters.
    - ``criterion``: the value of the criterion whose minimizer determines the frequency cutoff.
    - ``ll``: the log likelihood.
    - ``pars_init``: the initial guess of the parameters.
    - ``pars``: the optimized parameters.
    - ``thetas``: scale parameters for the gamma distribution.
    - ``timestep``: the time step.
    """
    # Maximize likelihood
    model.precondition(timestep, amplitudes[:nfit])
    pars_init = model.guess(freqs[:nfit], amplitudes[:nfit], ndofs[:nfit])
    if not model.valid(pars_init):
        raise AssertionError("Infeasible guess")
    cost = LowFreqCost(timestep, freqs[:nfit], amplitudes[:nfit], ndofs[:nfit], model)
    opt = minimize(
        cost.funcgrad,
        pars_init,
        jac=True,
        hess=cost.hess,
        bounds=model.bounds(),
        method="trust-constr",
        options={"xtol": 1e-10, "gtol": 1e-10},
    )
    props = cost.props(opt.x, 2)

    # Compute the Hessian and its properties.
    evals, evecs = np.linalg.eigh(props["cost_hess"])
    props["cost_hess_evals"] = evals
    props["cost_hess_evecs"] = evecs
    if (evals > 0).all() and np.isfinite(evals).all():
        half = evecs / np.sqrt(evals)
        props["covar"] = np.dot(half, half.T)
    else:
        props["covar"] = np.full_like(props["cost_hess"], np.inf)

    # Derive estimates from model parameters.
    props.update(model.derive_props(props["pars"], props["covar"]))

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
