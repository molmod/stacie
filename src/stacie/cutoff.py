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
"""Criteria for selecting the part of the spectrum to fit to."""

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.special import polygamma

from .model import SpectrumModel
from .spectrum import Spectrum
from .utils import PositiveDefiniteError, robust_posinv

__all__ = (
    "CV2LCriterion",
    "CutoffCriterion",
    "linear_weighted_regression",
    "switch_func",
)


def switch_func(x: NDArray[float], cutoff: float, exponent: float) -> NDArray[float]:
    """Evaluate the switching function at a given points x."""
    if np.any(x < 0):
        raise ValueError("x must be non-negative")
    return 1 / (1 + (x / cutoff) ** exponent)


WEIGHT_EPS = 1e-3


@attrs.define
class CutoffCriterion:
    """Base class for cutoff criteria.

    Subclasses should implement the ``__call__`` method.
    """

    @property
    def name(self) -> str:
        """The name of the criterion."""
        raise NotImplementedError

    def __call__(
        self,
        spectrum: Spectrum,
        model: SpectrumModel,
        props: dict[str, NDArray],
    ) -> dict[str, float]:
        """Compute the criterion for the given spectrum and model.

        Parameters
        ----------
        spectrum
            The spectrum object containing the input data.
        model
            The model to be fitted to the spectrum.
        props
            The property dictionary returned by the :py:meth:`stacie.cost.LowFreqCost.props` method.

        Returns
        -------
        results
            A dictionary with at least the following fields:

            - "criterion": minus the logarithm of a likelihood of "a good fit".
            - "criterion_expected": expected value of the negative log likelihood.
            - "criterion_var": expected variance of the negative log likelihood.
            - "msg": optional message explaining failure to compute the criterion.
            - "stop": optional flag to terminate the cutoff scan early.
        """
        raise NotImplementedError


@attrs.define
class CV2LCriterion(CutoffCriterion):
    """Criterion based on the difference between fits to two halves of the spectrum."""

    fcut_factor: float = attrs.field(default=1.25, kw_only=True)
    """The scale factor to apply to the cutoff frequency.

    If 1.0, the same part of the spectrum is used as in the full non-linear regression.
    By using a larger value, the default, the criterion also tests whether
    the fitted parameters can (somewhat) extrapolate to larger frequencies,
    which reduces the risk of underfitting.
    This results in less bias on the autocorrelation integral, but slightly larger variance.
    """

    log: bool = attrs.field(default=False, kw_only=True)
    """Whether to fit a linearized model to the logarithm of the spectrum."""

    cond: float = attrs.field(default=1e6, kw_only=True)
    """The threshold for the condition number of the preconditioned covariance matrix.

    Due to the preconditioning, the condition number should be close to 1.0.
    If not, the linear dependence of the parameters is too strong, making the fit unreliable.
    In this case, "inf" is returned as the criterion.
    """

    precondition: bool = attrs.field(default=True, kw_only=True)
    """Whether to precondition the covariance eigendecomposition.

    This option is only disabled for testing. Always leave it enabled in production.
    """

    regularize: bool = attrs.field(default=True, kw_only=True)
    """Whether to regularize the linear regression.

    This option is only disabled for testing. Always leave it enabled in production.
    It will only have an impact on very ill-conditioned fits.
    """

    @property
    def name(self) -> str:
        """The name of the criterion."""
        return f"cv2l({self.fcut_factor:.0%}{',log' if self.log else ''})"

    def __call__(
        self,
        spectrum: Spectrum,
        model: SpectrumModel,
        props: dict[str, NDArray],
    ) -> dict[str, float]:
        """The disparity between fits to two different parts of the spectrum."""
        # Compute weights for the two halves and the model
        fcut = props["fcut"]
        switch_exponent = props["switch_exponent"]
        freqs = spectrum.freqs
        weights = switch_func(freqs, self.fcut_factor * fcut, switch_exponent)
        weights1 = switch_func(freqs, 0.5 * self.fcut_factor * fcut, switch_exponent)
        weights2 = weights - weights1
        ncut = (weights > WEIGHT_EPS).sum()
        if ncut == len(freqs):
            return {
                "criterion": np.inf,
                "msg": "cv2l: Insufficient data after cutoff.",
                "stop": True,
            }
        freqs = freqs[:ncut]
        weights = weights[:ncut]
        weights1 = weights1[:ncut]
        weights2 = weights2[:ncut]
        amplitudes_model = model.compute(freqs, props["pars"], deriv=1)

        # Prepare the linear problem: transform to a basis where the covariance of
        # the non-linear regression becomes the identity matrix.
        alphas = 0.5 * spectrum.ndofs[:ncut]
        design_matrix = amplitudes_model[1].T
        if self.regularize:
            design_matrix /= props["cost_hess_scales"]
            design_matrix = np.dot(design_matrix, props["cost_hess_rescaled_evecs"])
            design_matrix /= props["cost_hess_rescaled_evals"] ** 0.5

        if self.log:
            # Construct a linear regression for the residual of the logarithm of the spectrum.
            design_matrix /= amplitudes_model[0][:, None]
            expected_values = (
                np.log(spectrum.amplitudes[:ncut])
                - polygamma(0, alphas)
                - np.log(amplitudes_model[0])
            )
            evstd = np.sqrt(polygamma(1, alphas))
        else:
            # Construct a linear regression for the residual of the spectrum.
            expected_values = spectrum.amplitudes[:ncut] - amplitudes_model[0]
            evstd = amplitudes_model[0] / np.sqrt(alphas)

        # Correct the standard deviation on the expected values for the fact
        # that they are residuals with fewer degrees of freedom than the original data.
        evstd *= np.sqrt((weights.sum() - model.npar) / weights.sum())

        # Transform equations to have unit variance on the expected values.
        design_matrix = design_matrix / evstd.reshape(-1, 1)
        expected_values = expected_values / evstd

        try:
            xs, cs = linear_weighted_regression(
                design_matrix,
                expected_values,
                np.array([weights1, weights2]),
                np.array([[1.0, -1.0]]),
                ridge=1e-6 if self.regularize else 0.0,
            )
        except ValueError as exc:
            return {
                "criterion": np.inf,
                "msg": f"cv2l: {exc.args[0]}",
            }
        xd = xs[0]
        cd = cs[0, :, 0]

        # Compute the difference between the two parameter vectors in the
        # basis of the covariance matrix of the difference
        if self.precondition:
            try:
                scales, evals, evecs, _ = robust_posinv(cd)
            except PositiveDefiniteError as exc:
                return {
                    "criterion": np.inf,
                    "msg": f"cv2l: Covariance {exc.args[0]}",
                }
            if evals.max() > self.cond * evals.min():
                return {
                    "criterion": np.inf,
                    "msg": f"cv2l: Linear dependencies in basis. {evals=}",
                }
        else:
            if not (np.isfinite(cd).all()):
                return {
                    "criterion": np.inf,
                    "msg": "cv2l: Covariance matrix is not finite.",
                }
            evals, evecs = np.linalg.eigh(cd)
            if evals.min() <= 0:
                return {
                    "criterion": np.inf,
                    "msg": "cv2l: Covariance matrix is not positive definite.",
                }
            scales = np.ones(len(evals))

        # Compute the negative log likelihood of the difference in parameters.
        delta = np.dot(evecs.T, xd / scales)
        criterion = (
            0.5 * (delta**2 / evals).sum()
            + 0.5 * np.log(2 * np.pi * evals).sum()
            + np.log(scales).sum()
        )
        expected = 0.5 * np.log(2 * np.pi * np.exp(1) * evals).sum() + np.log(scales).sum()
        variance = len(evals) / 2
        if self.regularize:
            penalty = -(
                np.log(props["cost_hess_scales"]).sum()
                + 0.5 * np.log(props["cost_hess_rescaled_evals"]).sum()
            )
            criterion += penalty
            expected += penalty

        return {"criterion": criterion, "criterion_expected": expected, "criterion_var": variance}


def linear_weighted_regression(
    dm: NDArray[float],
    ev: NDArray[float],
    ws: NDArray[float],
    lc: NDArray[float] | None = None,
    ridge: float = 0.0,
) -> tuple[NDArray[float], NDArray[float]]:
    """Perform a linear regression with multiple weight vectors.

    This is a helper function for cv2l_criterion.

    Parameters
    ----------
    dm
        The design matrix.
        Shape ``(neq, npar)``, where ``neq`` is the number of equations
        and ``npar`` is the number of parameters.
    ev
        The expected values, with standard normal measurement errors.
        Shape ``(neq,)``.
    ws
        A set of weight vectors for the rows of dm (equations).
        Shape ``(nw, neq)``, where ``nw`` is the number of weight vectors.
    lc
        Linear combinations of solutions for different weights to be computed.
        Shape ``(nlc, nw)``, where ``nlc`` is the number of linear combinations.
        If None, the identity matrix is used with shape ``(nw, nw)``.

    Returns
    -------
    xs
        The regression coefficients for each weight vector.
        Shape ``(nw, npar)``.
    cs
        The covariance matrices for each combination of weight vector.
        Shape ``(nw, npar, nw, npar)``.
    """
    # Precondition the design matrix by normalizing the columns.
    column_norms = np.linalg.norm(dm, axis=0)
    if np.any(column_norms == 0):
        raise ValueError("Design matrix has zero columns.")
    column_scales = 1 / column_norms
    dm = dm * column_scales
    # Perform SVD on each weighted design matrix.
    rws = np.sqrt(ws)
    u, s, vt = np.linalg.svd(np.einsum("we,ep->wep", rws, dm), full_matrices=False)
    sinv = s / (s**2 + ridge**2)
    # Solve the problem for each weight vector.
    xs = np.einsum("wip,wi,wei,e,we->wp", vt, sinv, u, ev, rws)
    # Construct the square roots of the covariance matrix for each weight vector.
    rcs = np.einsum("wip,wi,wei,we->wpe", vt, sinv, u, rws)
    # If lc is None, use the identity matrix.
    if lc is not None:
        # Work out the requested linear combinations.
        xs = np.einsum("lw,wp->lp", lc, xs)
        rcs = np.einsum("lw,wpe->lpe", lc, rcs)
    cs = np.einsum("wpe,vqe->wpvq", rcs, rcs)
    # Account for the preconditioning to get results for the original problem.
    xs = np.einsum("wp,p->wp", xs, column_scales)
    cs = np.einsum("wpvq,p,q->wpvq", cs, column_scales, column_scales)
    return xs, cs
