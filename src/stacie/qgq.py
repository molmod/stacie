"""Quasi-Gaussian Quadrature.

A hybrid method between Gaussian quadrature and Quasi-Monte Carlo integration.
Unlike Gaussian quadrature, the integrand is assumed to be noisy,
with independent and identically distributed noise.
Unlike Monte Carlo integration, the integration grid is fixed and guarantees
a certain level of accuracy for smooth integrands.
This module only handles 1D integrations.
"""

from collections.abc import Callable
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpy.polynomial.hermite_e import HermiteE
from numpy.typing import ArrayLike, NDArray

__all__ = ("construct_qgq_empirical", "construct_qgq_low", "construct_qgq_stdnormal", "plot_qgq")


def construct_qgq_stdnormal(
    points0: ArrayLike,
    nmoment: int,
    targets_weights: ArrayLike | None = None,
    weight_solver: Callable | None = None,
    eps: float = 1e-5,
    do_extra: bool = False,
) -> tuple[ArrayLike, ArrayLike] | tuple[ArrayLike, ArrayLike, dict]:
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
    weight_solver
        The solver used to assign the quadrature weights based on the current grid points.
        This can be either `solve_quadratic`, which minimizes the L2 norm of the weights,
        or `solve_kld`, which minimizes the negative entropy of the weights.
        if not given, `solve_kld` is used by default.
    eps
        A small regularization parameter to prevent points from collapsing or reshuffling.
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
    else:
        fix = []

    return construct_qgq_low(
        points0=points0,
        basis_funcs=[HermiteE.basis(i) for i in range(2, nmoment + 1, 2)],
        basis_funcs_d=[HermiteE.basis(i).deriv() for i in range(2, nmoment + 1, 2)],
        targets_basis=np.zeros(nmoment // 2),
        targets_weights=targets_weights,
        fix=fix,
        symmetric=True,
        weight_solver=weight_solver,
        eps=eps,
        do_extra=do_extra,
    )


def construct_qgq_empirical(
    samples: ArrayLike,
    points0: int | ArrayLike,
    nmoment: int,
    targets_weights: ArrayLike | None = None,
    weight_solver: Callable | None = None,
    eps: float = 1e-5,
    do_extra: bool = False,
) -> tuple[ArrayLike, ArrayLike] | tuple[ArrayLike, ArrayLike, dict]:
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
    weight_solver
        The solver used to assign the quadrature weights based on the current grid points.
        This can be either `solve_quadratic`, which minimizes the L2 norm of the weights,
        or `solve_kld`, which minimizes the negative entropy of the weights.
        if not given, `solve_kld` is used by default.
    eps
        A small regularization parameter to prevent points from collapsing or reshuffling.
    do_extra
        if True, also return an extra dictionary with additional information about the optimization,
        which can be used for debugging or analysis.

    Returns
    -------
    points
        The optimized grid points.
    weights
        The optimized quadrature weights, typically nearly proportional to the empirical weights.
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
        points0 = np.quantile(samples, (np.arange(npoint) + 0.5) / npoint)
    else:
        points0 = (np.asarray(points0, dtype=float) - mu) / sigma
    result = construct_qgq_low(
        points0=points0,
        basis_funcs=basis_funcs,
        basis_funcs_d=[basis_func.deriv() for basis_func in basis_funcs],
        targets_basis=targets_basis,
        targets_weights=targets_weights,
        fix=None,
        symmetric=False,
        weight_solver=weight_solver,
        eps=eps,
        do_extra=do_extra,
    )
    result[0][:] *= sigma
    result[0][:] += mu
    if do_extra:
        extra = result[2]
        extra["points0"] *= sigma
        extra["points0"] += mu
        extra["eqs0_d"] /= sigma
        extra["eqs1_d"] /= sigma
    return result


def construct_qgq_low(
    points0: ArrayLike,
    basis_funcs,
    basis_funcs_d,
    targets_basis: ArrayLike,
    targets_weights: ArrayLike | None = None,
    fix: ArrayLike | None = None,
    symmetric: bool = False,
    weight_solver: Callable | None = None,
    eps: float = 1e-5,
    do_extra: bool = False,
) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, dict[str]]:
    """Construct a quasi-Gaussian quadrature grid, low-level interface.

    Parameters
    ----------
    points0
        Initial grid points. The optimization starts from this grid,
        possibly keeping some points fixed as requested by the user.
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
        The target weights for the quadrature, which the quadrature should match.
        If not given, all values are set to 1.
    fix
        The indices of the points to keep fixed during optimization.
    weight_solver
        The solver used to assign the quadrature weights based on the current grid points.
        This can be either `solve_quadratic`, which minimizes the L2 norm of the weights,
        or `solve_kld`, which minimizes the negative entropy of the weights.
        if not given, `solve_kld` is used by default.
    symmetric
        If True, it is assumed that only the positive points are given in `points0`,
        and the negative ones must be added implicitly by symmetry.
    eps
        A small regularization parameter to prevent points from collapsing or reshuffling.
    do_extra
        if True, also return an extra dictionary with additional information about the optimization,
        which can be used for debugging or analysis.

    Returns
    -------
    points1
        The optimized grid points.
    weights
        The optimized quadrature weights, typically nearly proportional to the target weights.
    extra
        If `do_extra` is True, a dictionary containing additional information:

        - `wref`: The reference weights used for the optimization.
        - `points0`: The initial grid points.
        - `weights0`: The initial quadrature weights, based on the initial grid points.
        - `cost0`: The initial cost function value, which the optimization started from.
        - `grad0`: The initial gradient of the cost function with respect to the grid points
        - `eqs0`: The initial values of the basis functions at the initial grid points.
        - `eqs0_d`: The initial values of the derivatives of the basis functions.
        - `points1`: The optimized grid points, same as the first return value.
        - `weights1`: The optimized quadrature weights, same as the second return value.
        - `cost1`: The final cost function value, which the optimization minimized.
        - `grad1`: The final gradient of the cost function with respect to the grid points,
        - `eqs1`: The final values of the basis functions at the optimized grid points.
        - `eqs1_d`: The final values of the derivatives of the basis functions.
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
    if weight_solver is None:
        weight_solver = solve_kld
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
                weight_solver=weight_solver,
                symmetric=symmetric,
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
            weight_solver=weight_solver,
            symmetric=symmetric,
            eps=eps,
        )

    initial_tr_radius = np.diff(points0).min() / 10
    res = sp.optimize.minimize(
        cost_wrapper,
        points0[mask] if len(fix) > 0 else points0,
        method="trust-constr",
        jac=True,
        options={"maxiter": 1000, "gtol": 1e-15, "initial_tr_radius": initial_tr_radius},
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    if len(fix) > 0:
        points1 = points0.copy()
        points1[mask] = res.x
    else:
        points1 = res.x
    targets_basis = np.concatenate(([1], targets_basis))
    if symmetric:
        points0, _ = apply_symmetry(points0, targets_weights)
        points1, targets_weights = apply_symmetry(points1, targets_weights)
    eqs0, eqs0_d = build_eqs(points0, basis_funcs, basis_funcs_d)
    cost0, weights0, grad0 = weight_solver(eqs0, eqs0_d, targets_basis, targets_weights)
    eqs1, eqs1_d = build_eqs(points1, basis_funcs, basis_funcs_d)
    cost1, weights1, grad1 = weight_solver(eqs1, eqs1_d, targets_basis, targets_weights)
    if do_extra:
        extra = {
            "wref": targets_weights,
            "points0": points0,
            "weights0": weights0,
            "cost0": cost0,
            "grad0": grad0,
            "eqs0": eqs0,
            "eqs0_d": eqs0_d,
            "points1": points1,
            "weights1": weights1,
            "cost1": cost1,
            "grad1": grad1,
            "eqs1": eqs1,
            "eqs1_d": eqs1_d,
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
        ax.plot(extra["points0"], extra["weights0"], "ko-", label="initial")
        ax.plot(extra["points1"], extra["weights1"], "ro-", label="optimized")
        ax.set_xlabel("x")
        ax.set_ylabel("weight")
        ax.legend()

    def plot_eqs(ax):
        for i, eq in enumerate(extra["eqs1"]):
            ax.plot(extra["points1"], eq, "o-", label=f"eq {i}")
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
    weight_solver: Callable,
    symmetric: bool = False,
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
        The target weights for the quadrature, which the quadrature should match.
    weight_solver
        The solver used to assign the quadrature weights based on the current grid points.
    symmetric
        If True, it is assumed that only the positive points are given in `points`,
        and the negative ones must be added implicitly by symmetry.
    eps
        A small regularization parameter to prevent points from collapsing or reshuffling.

    Returns
    -------
    cost
        The cost function value, which the optimization will minimize.
    grad
        The gradient of the cost function with respect to the grid points, used for optimization.
    """
    if symmetric:
        nhalf = len(points)
        points, targets_weights = apply_symmetry(points, targets_weights)

    # Compute the quadrature weights and cost for fixed points.
    target_basis = np.concatenate(([1], targets_basis))
    eqs, eqs_d = build_eqs(points, basis_funcs, basis_funcs_d)
    cost, _, grad = weight_solver(eqs, eqs_d, target_basis, targets_weights)

    # Add a minor penalty to prevent points from collapsing or reshuffling.
    penalties = np.exp(-np.diff(points))
    cost += eps * penalties.sum()
    penalty_grad = np.zeros_like(points)
    penalty_grad[:-1] += eps * penalties
    penalty_grad[1:] -= eps * penalties
    grad += penalty_grad

    if symmetric:
        grad_half = grad[-nhalf:].copy()
        grad_half = -grad[:nhalf][::-1]
        grad = grad_half

    return cost, grad


def apply_symmetry(points_half, targets_weights_half):
    """Mirror the points and weights around zero to enforce symmetry."""
    if points_half[0] < 0:
        raise ValueError("points must be non-negative when symmetric is True")
    if points_half[0] == 0:
        points = np.concatenate((-points_half[1:][::-1], points_half))
        targets_weights = np.zeros_like(points)
        targets_weights[-len(targets_weights_half) :] = targets_weights_half
        targets_weights[: len(targets_weights_half)] += targets_weights_half[::-1]
    else:
        points = np.concatenate((-points_half[::-1], points_half))
        targets_weights = np.concatenate((targets_weights_half[::-1], targets_weights_half))
    return points, targets_weights


def build_eqs(points, basis_funcs, basis_funcs_d):
    eqs = [basis_func(points) for basis_func in basis_funcs]
    eqs_d = [basis_func_d(points) for basis_func_d in basis_funcs_d]
    eqs.insert(0, np.ones_like(points))
    eqs_d.insert(0, np.zeros_like(points))
    return np.array(eqs), np.array(eqs_d)


def solve_quadratic(eqs, eqs_d, rhs, wref):
    """Optimize the quadrature weights minimizing the sum of squares weights divided by wref.

    Parameters
    ----------
    eqs
        The linear coefficients of the constraints to impose.
        Shape `(nconstraint, npoint)`.
    eqs_d
        The derivatives of the linear coefficients with respect to the points.
        Shape `(nconstraint, npoint)`.
    rhs
        The target values of the constraints to impose.
        Shape `(nconstraint,)`.
    wref
        The reference weights to aim for.
        Shape `(npoint,)`.

    Returns
    -------
    cost
        The cost function value, which the optimization minimized.
    weights
        The optimized quadrature weights.
    grad
        The gradient of the cost function with respect to the grid points.
    """
    u, s, vh = np.linalg.svd(eqs * wref, full_matrices=False)
    ratios = np.dot(vh.T, np.dot(u.T, rhs) / s)
    l1 = np.dot(u, np.dot(vh, ratios) / s)
    cost = 0.5 * np.dot(ratios, ratios)
    weights = ratios * wref
    grad = -np.einsum("i,ij,j->j", l1, eqs_d, weights)
    return cost, weights, grad


def solve_kld(eqs, eqs_d, rhs, wref):
    """Optimize the quadrature weights by minimizing their KL-divergence from the reference.

    See `solve_quadratic` for the API documentation, which is the same as this function.
    """
    l0 = np.ones(len(eqs))
    res = sp.optimize.root(kld_constr, l0, args=(eqs, rhs, wref), jac=True, tol=1e-15)
    l1 = res.x
    lnratios = np.dot(-l1, eqs)
    weights = wref * np.exp(lnratios)
    cost = np.dot(lnratios, weights)
    grad = np.einsum("i,ij,j->j", l1, eqs_d, weights)
    return cost, weights, grad


def kld_constr(lbda, eqs, rhs, wref):
    """Helper to solve the Langrange multipliers for the KL-divergence optimization.

    Parameters
    ----------
    lbda
        The current values of the Lagrange multipliers.
        Shape `(nconstraint,)`.
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
    errors
        The current values of the constraint errors, to be made zero.
    jac
        The Jacobian of the constraint errors with respect to the Lagrange multipliers.
    """
    weights = wref * np.exp(np.dot(-lbda, eqs))
    errors = np.dot(eqs, weights) - rhs
    jac = -np.einsum("ij,j,kj->ik", eqs, weights, eqs)
    return errors, jac
