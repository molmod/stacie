"""Quasi-Gaussian Quadrature.

A hybrid method between Gaussian quadrature and Quasi-Monte Carlo integration.
Unlike Gaussian quadrature, the integrand is assumed to be noisy,
with independent and identically distributed noise.
Unlike Monte Carlo integration, the integration grid is fixed and guarantees
a certain level of accuracy for smooth integrands.
This module only handles 1D integrations.
"""

from collections.abc import Callable
from enum import Enum
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpy.polynomial.hermite_e import HermiteE
from numpy.typing import ArrayLike, NDArray

__all__ = ("construct_qgq_empirical", "construct_qgq_low", "construct_qgq_stdnormal", "plot_qgq")


class Symmetry(Enum):
    """Symmetry of the quadrature grid points (and weights) around zero."""

    # Enforce no symmetry
    NONE = 0

    # Even symmetry including x=0 fixed
    ZERO = 1

    # Even symmetry excluding x=0
    NONZERO = 2


def construct_qgq_stdnormal(
    points0: ArrayLike,
    nmoment: int,
    targets_weights: ArrayLike | None = None,
    gtol: float = 1e-13,
    eps: float = 1e-5,
    do_extra: bool = False,
) -> NDArray | tuple[NDArray, dict[str]]:
    """Construct a quasi-Gaussian quadrature grid for the standard normal distribution.

    Parameters
    ----------
    points0
        Initial grid points. The optimization starts from this grid,
        possibly keeping some x=0 fixed if present.
        Only provide positive points, as the negative ones will be automatically added by symmetry.
    nmoment
        The number of moments to match, must be strictly positive and even.
        Only even moments are considered, as the distribution is symmetric.
    targets_weights
        The target weights for the quadrature, which the quadrature should match.
        If not given, all values are set to 1, except when points0[0] = 0,
        in which case the first weight is set to 0.5 to account for the symmetry.
    gtol
        The gradient tolerance for the optimization convergence.
        The default is very strict.
    eps
        A small regularization parameter to prevent points from collapsing or reshuffling.
        Smaller values result in more even weights and slower optimization.
    do_extra
        if True, also return an extra dictionary with additional information about the optimization,
        which can be used for debugging or analysis.

    Returns
    -------
    points
        The optimized grid points, symmetric around zero.
    weights
        The optimized quadrature weights, typically nearly proportional to the Gaussian weights.
    extra
        If `do_extra` is True, a dictionary containing additional information.
    """
    if nmoment <= 0 or nmoment % 2 != 0:
        raise ValueError("nmoment must be strictly positive and even")
    if targets_weights is None:
        targets_weights = np.ones(len(points0))
    if points0[0] == 0:
        fix = [0]
        targets_weights[0] = 0.5
        symmetry = Symmetry.ZERO
    else:
        fix = []
        symmetry = Symmetry.NONZERO

    return construct_qgq_low(
        points0=points0,
        basis_funcs=[HermiteE.basis(i) for i in range(2, nmoment + 1, 2)],
        basis_funcs_d=[HermiteE.basis(i).deriv() for i in range(2, nmoment + 1, 2)],
        targets_basis=np.zeros(nmoment // 2),
        targets_weights=targets_weights,
        fix=fix,
        symmetry=symmetry,
        gtol=gtol,
        eps=eps,
        do_extra=do_extra,
    )


def construct_qgq_empirical(
    samples: ArrayLike,
    points0: int | ArrayLike,
    nmoment: int,
    targets_weights: ArrayLike | None = None,
    symmetry: Symmetry = Symmetry.NONE,
    gtol: float = 1e-13,
    eps: float = 1e-5,
    do_extra: bool = False,
) -> NDArray | tuple[NDArray, dict[str]]:
    """Construct a quasi-Gaussian quadrature grid for an empirical distribution.

    Parameters
    ----------
    samples
        The samples from the empirical distribution, used to compute the target moments.
    points0
        The number of grid points to optimize or the initial grid points.
    nmoment
        The number of moments to match,
        must be strictly positive and strictly less than npoint.
    targets_weights
        The target weights for the quadrature, which the quadrature should match.
        If not given, all values are set to 1.
    symmetry
        The symmetry of the quadrature grid points (and weights) around zero.
        If symmetry is imposed, only the positive points are given in `points0`,
        and the negative ones are added implicitly by symmetry.
    gtol
        The gradient tolerance for the optimization convergence.
        The default is very strict.
    eps
        A small regularization parameter to prevent points from collapsing or reshuffling.
        Smaller values result in more even weights and slower optimization.
    do_extra
        if True, also return an extra dictionary with additional information about the optimization,
        which can be used for debugging or analysis.

    Returns
    -------
    points
        The optimized grid points.
    weights
        The optimized quadrature weights, typically nearly proportional to the Gaussian weights.
    extra
        If `do_extra` is True, a dictionary containing additional information.
    """
    if isinstance(points0, int):
        npoint = points0
        points0 = None
    else:
        npoint = len(points0)
    if nmoment <= 0:
        raise ValueError("nmoment must be strictly positive")
    if nmoment >= npoint:
        raise ValueError("nmoment must be less than npoint")
    samples = np.asarray(samples, dtype=float)
    mu = samples.mean()
    sigma = samples.std()
    if sigma <= 0:
        raise ValueError("samples must have positive standard deviation")
    samples = (samples - mu) / sigma
    basis_funcs = [HermiteE.basis(i) for i in range(1, nmoment + 1, 1)]
    targets_basis = [basis_func(samples).mean() for basis_func in basis_funcs]
    if points0 is None:
        if symmetry == Symmetry.NONE:
            qs = (np.arange(npoint) + 0.5) / npoint
        else:
            frac_neg = (samples < 0).mean()
            frac_pos = 1 - frac_neg
            if symmetry == Symmetry.ZERO:
                if npoint % 2 == 0:
                    raise ValueError("npoint must be odd when symmetry is ZERO")
                nhalf = npoint // 2 + 1
                qs = np.arange(nhalf) / nhalf
            else:
                if npoint % 2 != 0:
                    raise ValueError("npoint must be even when symmetry is NONZERO")
                nhalf = npoint // 2
                qs = (np.arange(nhalf) + 0.5) / nhalf
            qs = frac_neg + qs * frac_pos
        points0 = np.quantile(samples, qs)
    else:
        points0 = (np.asarray(points0, dtype=float) - mu) / sigma

    result = construct_qgq_low(
        points0=points0,
        basis_funcs=basis_funcs,
        basis_funcs_d=[basis_func.deriv() for basis_func in basis_funcs],
        targets_basis=targets_basis,
        targets_weights=targets_weights,
        fix=None,
        symmetry=symmetry,
        gtol=gtol,
        eps=eps,
        do_extra=do_extra,
    )
    points = result[0]
    if do_extra:
        extra = result[-1]
        extra["points0"] *= sigma
        extra["points0"] += mu
        extra["eqs0_d"] /= sigma
        extra["eqs1_d"] /= sigma
    points[:] *= sigma
    points[:] += mu
    return result


def construct_qgq_low(
    points0: ArrayLike,
    basis_funcs,
    basis_funcs_d,
    targets_basis: ArrayLike,
    targets_weights: ArrayLike | None = None,
    fix: ArrayLike | None = None,
    symmetry: Symmetry = Symmetry.NONE,
    gtol: float = 1e-13,
    eps: float = 1e-5,
    do_extra: bool = False,
) -> NDArray | tuple[NDArray, dict[str]]:
    """Construct a quasi-Gaussian quadrature grid, low-level interface.

    Parameters
    ----------
    points0
        Initial grid points. The optimization starts from this grid,
        possibly keeping some points fixed as requested by the user.

        Note that the algorithm assumes that the mean of the distribution
        is well-behaved, not large compared to 1,
        and that the spread is of the order 1.
        If this is not the case, it is recommended to precondition the
        problem by centering and scaling the points.
        See `construct_qgq_empirical` for an example of how to do this.
    basis_funcs
        The basis functions whose integrals are to be matched by the quadrature.
        These are functions of a single variable, and are evaluated at the grid points.
        They must be able to vectorize over the grid points.
        Note that a constant basis function is automatically added to the list of basis functions.
    basis_funcs_d
        The derivatives of the basis functions,
        used for computing the gradient of the cost function.
    targets_basis
        The target integrals of the basis functions, which the quadrature should match.
    targets_weights
        The target weights for the quadrature points.
    fix
        The indices of the points to keep fixed during optimization.
    symmetry
        The symmetry of the quadrature grid points (and weights) around zero.
        If symmetry is imposed, only the positive points are given in `points0`,
        and the negative ones are added implicitly by symmetry.
    gtol
        The gradient tolerance for the optimization convergence.
        The default is very strict.
    eps
        A small regularization parameter to prevent points from collapsing or reshuffling.
        Smaller values result in more even weights and slower optimization.
    do_extra
        if True, also return an extra dictionary with additional information about the optimization,
        which can be used for debugging or analysis.

    Returns
    -------
    points1
        The optimized grid points.
    weights1
        The optimized quadrature weights.
    extra
        If `do_extra` is True, a dictionary containing intermediate results:

        - `points0`: The initial grid points.
        - `weights0`: The initial quadrature weights.
        - `eqs0`: The initial values of the basis functions at the initial grid points.
        - `eqs0_d`: The initial values of the derivatives of the basis functions.
        - `cost0`: The initial cost function value.
        - `grad0`: The initial gradient of the cost function with respect to the grid points
        - `points1`: The optimized grid points, same as the first return value.
        - `weights1`: The optimized quadrature weights.
        - `eqs1`: The final values of the basis functions at the optimized grid points.
        - `eqs1_d`: The final values of the derivatives of the basis functions.
        - `cost1`: The final cost function value.
        - `grad1`: The final gradient of the cost function with respect to the grid points
    """
    points0 = np.asarray(points0, dtype=float)
    fix = np.array([], dtype=int) if fix is None else np.asarray(fix, dtype=int)
    targets_basis = np.asarray(targets_basis, dtype=float)
    if len(targets_basis) != len(basis_funcs):
        raise ValueError("targets_basis must have the same length as basis_funcs")
    if len(targets_basis) != len(basis_funcs_d):
        raise ValueError("targets_basis must have the same length as basis_funcs_d")
    if len(targets_basis) >= len(points0):
        raise ValueError("the number of basis functions must be less than the number of points")
    if targets_weights is None:
        targets_weights = np.ones(len(points0))
    else:
        targets_weights = np.asarray(targets_weights, dtype=float)
        if targets_weights.min() <= 0:
            raise ValueError("targets_weights must be strictly positive")

    if np.diff(points0).min() <= 0:
        raise ValueError("points0 must be strictly increasing")
    if symmetry == Symmetry.ZERO and points0[0] != 0:
        raise ValueError("points0[0] must be 0 when symmetry is ZERO")
    if symmetry == Symmetry.NONZERO and np.any(points0 <= 0):
        raise ValueError("all points0 must be strictly positive when symmetry is NONZERO")
    if eps < 0:
        raise ValueError("eps must be positive")

    if len(fix) > 0:
        mask = np.ones(len(points0), dtype=bool)
        mask[fix] = False

        def cost_wrapper(points_small):
            points = points0.copy()
            points[mask] = points_small
            cost, grad = point_cost(
                points,
                basis_funcs=basis_funcs,
                basis_funcs_d=basis_funcs_d,
                targets_basis=targets_basis,
                targets_weights=targets_weights,
                symmetry=symmetry,
                eps=eps,
            )
            return cost, grad[mask]
    else:
        cost_wrapper = partial(
            point_cost,
            basis_funcs=basis_funcs,
            basis_funcs_d=basis_funcs_d,
            targets_basis=targets_basis,
            targets_weights=targets_weights,
            symmetry=symmetry,
            eps=eps,
        )

    res = sp.optimize.minimize(
        cost_wrapper,
        points0[mask] if len(fix) > 0 else points0,
        method="L-BFGS-B",
        options={
            "gtol": gtol,
            "ftol": 0,
            "maxiter": 100 * len(points0) ** 2,
            "maxfun": 100 * len(points0) ** 2,
        },
        jac=True,
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    if len(fix) > 0:
        points1 = points0.copy()
        points1[mask] = res.x
    else:
        points1 = res.x
    if symmetry != Symmetry.NONE:
        points0 = apply_symmetry_points(points0, symmetry == Symmetry.ZERO)
        points1 = apply_symmetry_points(points1, symmetry == Symmetry.ZERO)
        targets_weights = apply_symmetry_weights(targets_weights, symmetry == Symmetry.ZERO)
    eqs0, eqs0_d = build_eqs(points0, basis_funcs, basis_funcs_d)
    eqs1, eqs1_d = build_eqs(points1, basis_funcs, basis_funcs_d)
    weights0 = assign_weights_l2(eqs0, targets_basis, targets_weights)
    weights1 = assign_weights_l2(eqs1, targets_basis, targets_weights)
    order = np.argsort(points1)
    points1 = points1[order]
    weights1 = weights1[order]
    cost0, grad0 = point_cost(
        points0,
        basis_funcs,
        basis_funcs_d,
        targets_basis,
        targets_weights,
        symmetry=symmetry,
        eps=eps,
    )
    cost1, grad1 = point_cost(
        points1,
        basis_funcs,
        basis_funcs_d,
        targets_basis,
        targets_weights,
        symmetry=symmetry,
        eps=eps,
    )
    if do_extra:
        extra = {
            "points0": points0,
            "weights0": weights0,
            "eqs0": eqs0,
            "eqs0_d": eqs0_d,
            "cost0": cost0,
            "grad0": grad0,
            "points1": points1,
            "weights1": weights1,
            "eqs1": eqs1,
            "eqs1_d": eqs1_d,
            "cost1": cost1,
            "grad1": grad1,
        }
        return points1, weights1, extra
    return points1, weights1


def plot_qgq(
    extra: dict[str], plot_dist: Callable | None = None, figsize: tuple[int, int] | None = None
):
    """Plot the results of the QGQ optimization."""
    if plot_dist is None:
        if figsize is None:
            figsize = (4, 5)
        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        if figsize is None:
            figsize = (4, 7.5)
        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
        plot_dist(axs[0])

    def plot_points(ax):
        ax.plot(extra["points0"], extra["weights0"], "k+-", label="initial")
        ax.plot(extra["points1"], extra["weights1"], "r+-", label="optimized")
        ax.set_xlabel("x")
        ax.set_ylabel("weight")
        ax.legend()

    def plot_eqs(ax):
        for i, eq in enumerate(extra["eqs1"]):
            ax.plot(extra["points1"], eq, "+-", label=f"eq {i}")
        ax.set_xlabel("x")
        ax.set_ylabel("basis functions")

    plot_points(axs[-2])
    plot_eqs(axs[-1])
    return fig, axs


def point_cost(
    points: NDArray,
    basis_funcs: list,
    basis_funcs_d: list,
    targets_basis: NDArray,
    targets_weights: NDArray,
    symmetry: Symmetry = Symmetry.NONE,
    eps: float = 1e-5,
):
    """The cost function for optimizing the quadrature points.

    Parameters
    ----------
    points
        The current grid points.
    basis_funcs
        The basis functions whose integrals are to be matched by the quadrature.
    basis_funcs_d
        The derivatives of the basis functions, used for computing the gradient of the cost.
    targets_basis
        The target integrals of the basis functions, which the quadrature should match.
    targets_weights
        The target weights for the quadrature points.
    symmetry
        The symmetry of the quadrature grid points (and weights) around zero.
        If symmetry is imposed, only the positive points are given in `points0`,
        and the negative ones are added implicitly by symmetry.
    eps
        A small regularization parameter to prevent points from collapsing or reshuffling.

    Returns
    -------
    cost
        The cost function value, which the optimization will minimize.
    grad
        The gradient of the cost function with respect to the grid points, used for optimization.
    """
    if symmetry != Symmetry.NONE:
        nhalf = len(points)
        points = apply_symmetry_points(points, symmetry == Symmetry.ZERO)
        targets_weights = apply_symmetry_weights(targets_weights, symmetry == Symmetry.ZERO)

    # Compute the quadrature weights and cost for fixed points.
    eqs, eqs_d = build_eqs(points, basis_funcs, basis_funcs_d)
    targets_weights = targets_weights / targets_weights.sum()
    errors = np.dot(eqs, targets_weights) - targets_basis
    cost = 0.5 * np.dot(errors, errors)
    grad = np.dot(errors, eqs_d) * targets_weights

    # Add a minor penalty to prevent points from collapsing or reshuffling.
    # Here, we assume that the scale of the points is of the order 1.
    penalties = eps * np.exp(-np.diff(points))
    cost += penalties.sum()
    penalty_grad = np.zeros_like(points)
    penalty_grad[:-1] += penalties
    penalty_grad[1:] -= penalties
    grad += penalty_grad

    if symmetry != Symmetry.NONE:
        grad_half = grad[-nhalf:].copy()
        grad_half = -grad[:nhalf][::-1]
        grad = grad_half

    return cost, grad


def apply_symmetry_points(points_half: NDArray, zero: bool) -> NDArray:
    """Mirror the points and weights around zero to enforce symmetry."""
    if zero:
        points = np.concatenate((-points_half[1:][::-1], points_half))
    else:
        points = np.concatenate((-points_half[::-1], points_half))
    return points


def apply_symmetry_weights(weights_half: NDArray, zero: bool) -> NDArray:
    """Mirror the weights around zero to enforce symmetry."""
    if zero:
        weights = np.zeros(2 * len(weights_half) - 1)
        weights[: len(weights_half)] = weights_half[::-1]
        weights[-len(weights_half) :] += weights_half
    else:
        weights = np.concatenate((weights_half[::-1], weights_half))
    return weights


def build_eqs(points, basis_funcs, basis_funcs_d):
    eqs = [basis_func(points) for basis_func in basis_funcs]
    eqs_d = [basis_func_d(points) for basis_func_d in basis_funcs_d]
    return np.array(eqs), np.array(eqs_d)


def assign_weights_l2(eqs, rhs, wref):
    """Assign quadrature weights by solving the linear system eqs @ weights = rhs.

    Parameters
    ----------
    eqs
        The linear coefficients of the constraints to impose.
        Shape `(nconstraint, npoint)`.
    rhs
        The target values of the constraints to impose.
        Shape `(nconstraint,)`.
    wref
        The reference weights to aim for.
        Shape `(npoint,)`.

    Returns
    -------
    weights
        The assigned quadrature weights.
    """
    eqs, rhs = extend_eqs(eqs, rhs)
    u, s, vh = np.linalg.svd(eqs * wref, full_matrices=False)
    return np.dot(vh.T, np.dot(u.T, rhs) / s) * wref


def extend_eqs(eqs, rhs):
    """Add the mandatory constraint that the sum of weights equals 1."""
    eqs = np.concatenate([np.ones((1, eqs.shape[1])), eqs], axis=0)
    rhs = np.concatenate([[1], rhs])
    return eqs, rhs
