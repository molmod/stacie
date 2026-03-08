"""Designed Quadrature (DQ).

This DQ implementation assumes fixed quadrature weights, by default equal weights
as in Chebyshev quadrature, but the user can specify any positive weights.
Unlike standard Gaussian quadrature, the integration grids are optimal for noisy integrands,
for which the main source of error is the variance of the integrand rather than the
limited polynomial degree of the quadrature.
This is achieved by optimizing the grid point positions assuming equal (or fixed) weights.
Equal weights minimize the variance of the numerical integral when all function values on the grid
have independent and identically distributed noise.
Unlike Monte Carlo integration, the integration grid is deterministic and guarantees
a certain level of polynomial degree for smooth integrands.
The main contribution of this module is to provide a practical and efficient implementation
for the optimization of the grid points.
It only supports 1D grids.
This module also offers a few additional features that may come in handy for practical applications:

- One can specify fixed non-equal weights for the quadrature.
- One can impose symmetry on the weights, such that uneven integrands are integrated exactly.
"""

from collections.abc import Callable
from enum import Enum

import attrs
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite_e import HermiteE
from numpy.typing import ArrayLike, NDArray

__all__ = (
    "Equations",
    "Symmetry",
    "construct_dq_empirical",
    "construct_dq_low",
    "construct_dq_stdnormal",
    "dq3",
    "plot_dq",
)


class Symmetry(Enum):
    """Symmetry of the quadrature grid points (and weights) around zero."""

    # Enforce no symmetry
    NONE = 0

    # Even symmetry including x=0 fixed
    ZERO = 1

    # Even symmetry excluding x=0
    NONZERO = 2


def construct_dq_stdnormal(
    points0: ArrayLike,
    nmoment: int,
    symmetry: Symmetry = Symmetry.ZERO,
    weights0: ArrayLike | None = None,
    rmsdtol: float = 1e-14,
    maxiter: int = 1000,
    maxridge: int = 100,
    verbose: bool = False,
    do_extra: bool = False,
) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, dict[str]]:
    """Construct a DQ grid for the standard normal distribution.

    Parameters
    ----------
    points0
        The optimization starts from this grid.
        If symmetry is imposed, only the positive points should be given in `points0`.
        If symmetry is ZERO, the zero point is implicitly added,
        and should not be included in `points0`.
    nmoment
        The number of moments to match, must be strictly positive and even.
        Only even moments are considered, as the distribution is symmetric.
    symmetry
        The symmetry of the quadrature grid points (and weights) around zero.
        For the standard normal distribution, symmetry must be imposed.
        One can choose whether to include the zero point in the grid or not.
    weights0
        The desired weights for the quadrature.
        If not given, all weights are set equal.
        If symmetry is ZERO, the weight of the zero point must be included.
    rmsdtol
        The convergence threshold for the RMSD of the equations being solved iteratively.
        The default is very strict.
    maxiter
        The maximum number of iterations for the optimization.
    maxridge
        The maximum number of ridge adjustments for each iteration of the optimization.
    verbose
        If True, print the optimization progress at each iteration.
    do_extra
        if True, also return an extra dictionary with additional information about the optimization,
        which can be used for debugging or analysis.

    Returns
    -------
    points1
        The optimized grid points, symmetric around zero.
    weights1
        The optimized quadrature weights, typically matching the given weights.
    extra
        If `do_extra` is True, a dictionary containing additional information.
    """
    if symmetry == Symmetry.NONE:
        raise ValueError("Symmetry must be imposed for the standard normal distribution")
    if nmoment <= 0 or nmoment % 2 != 0:
        raise ValueError("nmoment must be strictly positive and even")

    funcs = [HermiteE.basis(i) for i in range(2, nmoment + 1, 2)]
    funcs_d = [func.deriv() for func in funcs]
    funcs_dd = [func.deriv() for func in funcs_d]

    return construct_dq_low(
        points0=points0,
        funcs=funcs,
        funcs_d=funcs_d,
        funcs_dd=funcs_dd,
        targets=np.zeros(nmoment // 2),
        weights0=weights0,
        symmetry=symmetry,
        rmsdtol=rmsdtol,
        maxiter=maxiter,
        maxridge=maxridge,
        verbose=verbose,
        do_extra=do_extra,
    )


def construct_dq_empirical(
    samples: ArrayLike,
    points0: int | ArrayLike,
    nmoment: int,
    symmetry: Symmetry = Symmetry.NONE,
    weights0: ArrayLike | None = None,
    rmsdtol: float = 1e-14,
    maxiter: int = 1000,
    maxridge: int = 100,
    verbose: bool = False,
    do_extra: bool = False,
) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, dict[str]]:
    """Construct a DQ grid for an empirical distribution.

    Parameters
    ----------
    samples
        The samples from the empirical distribution, used to compute the target moments.
    points0
        The number of grid points to optimize or the initial grid points.
        If symmetry is imposed, only the positive points should be given in `points0`.
        If symmetry is ZERO, the zero point is implicitly added,
        and should not be included in `points0`.
    nmoment
        The number of moments to match,
        must be strictly positive and strictly less than npoint.
    symmetry
        The symmetry of the quadrature grid points (and weights) around zero.
        If symmetry is imposed, only the positive points are given in `points0`,
        and the negative ones are added implicitly by symmetry.
    weights0
        The desired weights for the quadrature, which will be normalized to sum to 1.
        If not given, all weights are set equal.
        If symmetry is ZERO, the weight of the zero point must be included.
    rmsdtol
        The convergence threshold for the RMSD of the equations being solved iteratively.
        The default is very strict.
    maxiter
        The maximum number of iterations for the optimization.
    maxridge
        The maximum number of ridge adjustments for each iteration of the optimization.
    verbose
        If True, print the optimization progress at each iteration.
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
    funcs = [HermiteE.basis(i) for i in range(1, nmoment + 1, 1)]
    funcs_d = [func.deriv() for func in funcs]
    funcs_dd = [func.deriv() for func in funcs_d]
    targets = [func(samples).mean() for func in funcs]
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

    result = construct_dq_low(
        points0=points0,
        funcs=funcs,
        funcs_d=funcs_d,
        funcs_dd=funcs_dd,
        targets=targets,
        symmetry=symmetry,
        weights0=weights0,
        rmsdtol=rmsdtol,
        maxiter=maxiter,
        maxridge=maxridge,
        verbose=verbose,
        do_extra=do_extra,
    )
    points = result[0]
    points[:] *= sigma
    points[:] += mu
    if do_extra:
        extra = result[2]
        extra["points0"][:] *= sigma
        extra["points0"][:] += mu
    return result


def construct_dq_low(
    points0: ArrayLike,
    funcs: list[Callable],
    funcs_d: list[Callable],
    funcs_dd: list[Callable],
    targets: ArrayLike,
    symmetry: Symmetry = Symmetry.NONE,
    weights0: ArrayLike | None = None,
    rmsdtol: float = 1e-14,
    maxiter: int = 1000,
    maxridge: int = 100,
    verbose: bool = False,
    do_extra: bool = False,
) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, dict[str]]:
    """Construct a DQ grid, low-level interface.

    Parameters
    ----------
    points0
        Initial grid points. The optimization starts from this grid.
        If symmetry is imposed, only the positive points should be given in `points0`.
        If symmetry is ZERO, the zero point is implicitly added,
        and should not be included in `points0`.

        Note that the algorithm assumes that the optimal points are pre-conditioned
        such that they have mean zero and standard deviation one.
        See `construct_dq_empirical` for an example of how to do this.
    funcs
        The basis functions whose integrals are to be matched by the quadrature.
        These are functions of a single variable, and are evaluated at the grid points.
        They must be able to vectorize over the grid points.
        Note that a constant basis function is automatically added to the list of basis functions.
    funcs_d
        The derivatives of the basis functions.
    funcs_dd
        The second derivatives of the basis functions.
    targets
        The target integrals of the basis functions, which the quadrature should match.
    symmetry
        The symmetry of the quadrature grid points (and weights) around zero.
        If symmetry is imposed, only the positive points are given in `points0`,
        and the negative ones are added implicitly by symmetry.
    weights0
        The target weights for the quadrature points.
        If not given, all weights are set equal.
        If symmetry is ZERO, the weight of the zero point must be included.
    rmsdtol
        The convergence threshold for the RMSD of the equations being solved iteratively.
        The default is very strict.
    maxiter
        The maximum number of iterations for the optimization.
    maxridge
        The maximum number of ridge adjustments for each iteration of the optimization.
    verbose
        If True, print the optimization progress at each iteration.
    do_extra
        if True, also return an extra dictionary with additional information about the optimization,
        which can be used for debugging or analysis.

    Returns
    -------
    points1
        The optimized grid points.
        If symmetry is imposed, the full grid is returned, including the negative points.
    weights1
        The optimized quadrature weights corresponding to the optimized grid points.
        These weights are typically nearly proportional to the given weights.
    extra
        If `do_extra` is True, a dictionary containing intermediate results:

        - `points0`: The initial grid points.
        - `weights0`: The initial quadrature weights.
        - `errors0`: The errors of the equations at the initial grid points and weights.
        - `jacobian0`: The Jacobian of the equations at the initial grid points and weights.
        - `funcs0`: The basis functions evaluated at the initial grid points.
        - `points1`: The optimized grid points, same as the first return value.
        - `weights1`: The optimized quadrature weights.
        - `errors1`: The errors of the equations at the optimized grid points and weights.
        - `jacobian1`: The Jacobian of the equations at the optimized grid points and weights.
        - `funcs1`: The basis functions evaluated at the optimized grid points.
    """
    points0 = np.asarray(points0, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(targets) != len(funcs):
        raise ValueError("targets must have the same length as funcs")
    if len(targets) != len(funcs_d):
        raise ValueError("targets must have the same length as funcs_d")
    if len(targets) != len(funcs_dd):
        raise ValueError("targets must have the same length as funcs_dd")
    if len(targets) > len(points0):
        raise ValueError("the number of basis functions must not exceed the number of points")
    if weights0 is None:
        if symmetry == Symmetry.ZERO:
            weights0 = np.full(len(points0) + 1, 1 / (2 * len(points0) + 1))
        elif symmetry == Symmetry.NONZERO:
            weights0 = np.full(len(points0), 1 / (2 * len(points0)))
        else:
            weights0 = np.full(len(points0), 1 / len(points0))
    else:
        weights0 = np.asarray(weights0, dtype=float)
        if weights0.min() <= 0:
            raise ValueError("weights must be strictly positive")
        # Check length of weights
        if symmetry == Symmetry.ZERO:
            if len(weights0) != len(points0) + 1:
                raise ValueError("weights0 must have length len(points0) + 1 when symmetry is ZERO")
        elif len(weights0) != len(points0):
            raise ValueError("weights0 must have the same length as points0 when symmetry != ZERO")
        # Renormalize weights to sum to 1, taking into account the symmetry.
        if symmetry == Symmetry.ZERO:
            weights0_sum = weights0[0] + 2 * weights0[1:].sum()
        if symmetry == Symmetry.NONZERO:
            weights0_sum = 2 * weights0.sum()
        elif symmetry == Symmetry.NONE:
            weights0_sum = weights0.sum()
        weights0 = weights0 / weights0_sum

    eqs = Equations(
        funcs=funcs,
        funcs_d=funcs_d,
        funcs_dd=funcs_dd,
        targets=targets,
        weights0=weights0,
        symmetry=symmetry,
    )

    points1 = solve_modified_lm(points0, eqs, rmsdtol, maxiter, maxridge, verbose)

    # Apply symmetry to the final points and solve for the final weights.
    points1_sym, weights1_sym = eqs.apply_sym(points1)
    weights1_opt = eqs.solve_weights(points1_sym, weights1_sym)
    if do_extra:
        points0_sym, weights0_sym = eqs.apply_sym(points0)
        weights0_opt = eqs.solve_weights(points0_sym, weights0_sym)
        errors0, jacobian0 = eqs.compute_low(points0_sym, weights0_sym, deriv=1)
        funcs0 = np.array([func(points0_sym) for func in eqs.funcs])
        errors1, jacobian1 = eqs.compute_low(points1_sym, weights1_sym, deriv=1)
        funcs1 = np.array([func(points1_sym) for func in eqs.funcs])
        extra = {
            "points0": points0_sym,
            "weights0": weights0_opt,
            "errors0": errors0,
            "jacobian0": jacobian0,
            "funcs0": funcs0,
            "points1": points1_sym,
            "weights1": weights1_opt,
            "errors1": errors1,
            "jacobian1": jacobian1,
            "funcs1": funcs1,
        }
        return points1_sym, weights1_opt, extra
    return points1_sym, weights1_opt


@attrs.define
class Equations:
    """Implementation of the equations to solve for the DQ optimization.

    The system of equations contains obviously the constraints
    that the basis functions must be integrated correctly.
    Additionally, it contains additional equations that regularize the
    optimal points to be as uniformly spaced as possible.
    However, these extra equations are constructed such that their jacobian is
    always orthogonal to the jacobian of the basis function constraints.
    This ensures that the regularization does not affect the basis constraints.

    Optionally, symmetry can be imposed on the points.
    """

    funcs: list[Callable]
    funcs_d: list[Callable]
    funcs_dd: list[Callable]
    targets: NDArray
    weights0: NDArray
    symmetry: Symmetry

    def __call__(self, x, *, deriv=0):
        """Evaluate the equations to be zerod and optionally their Jacobian."""
        x2, w2 = self.apply_sym(x)
        result = self.compute_low(x2, w2, deriv=deriv)
        if deriv == 0:
            return result
        jac2 = result[1]
        jac = self.backprop_sym(jac2)
        if deriv == 1:
            return [result[0], jac]
        raise NotImplementedError("Derivatives beyond Jacobian not implemented")

    def apply_sym(self, x):
        """Return points and weights with their symmetric counterparts if symmetry is imposed."""
        w = self.weights0
        if self.symmetry == Symmetry.NONE:
            return x, w
        if self.symmetry == Symmetry.ZERO:
            return (
                np.concatenate((-x[::-1], [0], x)),
                np.concatenate((w[-1:0:-1], w)),
            )
        return (np.concatenate((-x[::-1], x)), np.concatenate((w[::-1], w)))

    def backprop_sym(self, jac):
        """Backpropagate the Jacobian through the symmetry operation."""
        if self.symmetry == Symmetry.NONE:
            return jac
        nindep = (jac.shape[1] - (self.symmetry == Symmetry.ZERO)) // 2
        return -jac[:, nindep - 1 :: -1] + jac[:, -nindep:]

    def compute_low(self, x, w, *, deriv=0):
        """Compute the equations and optionally their Jacobian for the given points and weights."""
        nf = len(self.funcs)
        nx = len(x)
        neq = nf if nf == nx else nf + nx
        eqs = np.zeros(neq)
        for i, func in enumerate(self.funcs):
            eqs[i] = np.dot(func(x), w) - self.targets[i]
        if nx > nf:
            proj_results = self.proj(x, w, deriv=deriv)
            pen_results = self.pen(x, w, deriv=deriv + 1)

            apply_proj = proj_results[1]
            grad = pen_results[1]
            gproj = apply_proj(grad)
            eqs[nf:] = gproj
        else:
            proj_results = self.proj(x, w, deriv=0)
        if deriv == 0:
            return (eqs,)
        jac = np.zeros((neq, nx))
        mat1 = proj_results[0]
        jac[:nf] = mat1
        if nx > nf:
            apply_proj_d = proj_results[2]
            hess = pen_results[2]
            jac[nf:] = apply_proj_d(grad) + apply_proj(hess)
        if deriv == 1:
            return eqs, jac
        raise NotImplementedError(f"Cannot compute for deriv={deriv}")

    def proj(self, x, w, *, deriv=0):
        r"""Construct projection on the orthogonal complement of the derivatives of the basis.

        Returns
        -------
        mat1
            Weighted derivative of basis functions w.r.t. x.
        apply_proj
            A function that applies the projection matrix to a vector:
            $P_{i,j}\, v_j$.
        apply_proj_d
            A function that applies the derivative of the projection matrix to a vector:
            $\frac{\mathrm{d}\,P_{i,j}}{\mathrm{d}\,x_k} v_j$.
        """
        mat1 = np.array([func(x) * w for func in self.funcs_d])
        u, s, vt = np.linalg.svd(mat1, full_matrices=False)

        def apply_proj(x):
            return x - np.dot(vt.T, np.dot(vt, x))

        if deriv == 0:
            return mat1, apply_proj

        mat1_pinv = np.dot(vt.T, (u / s).T)
        mat2 = np.array([func(x) * w for func in self.funcs_dd])
        pre_hess = np.dot(mat1_pinv, mat2)
        outer = np.dot(vt.T, vt)

        def apply_proj_d(x):
            y = np.dot(x, pre_hess)
            return outer * y - np.diag(y) - pre_hess * apply_proj(x)

        if deriv == 1:
            return mat1, apply_proj, apply_proj_d

        raise NotImplementedError(f"Cannot construct for deriv={deriv}")

    def pen(self, x, w, *, deriv=0):
        """Construct the regularization penalty and optionally its gradient and Hessian."""
        n = len(x)
        pen0 = -np.log(abs(np.diff(x)))
        w2 = np.sqrt(w[1:] * w[:-1]) / n
        func = np.dot(pen0, w2)
        if deriv == 0:
            return (func,)
        grad = np.zeros(n)
        pen1 = w2 / np.diff(x)
        grad[:-1] += pen1
        grad[1:] -= pen1
        if deriv == 1:
            return func, grad
        pen2 = w2 / (np.diff(x)) ** 2
        ia = np.arange(n - 1)
        ib = ia + 1
        hess = np.zeros((n, n))
        hess[ia, ia] += pen2
        hess[ia, ib] -= pen2
        hess[ib, ia] -= pen2
        hess[ib, ib] += pen2
        if deriv == 2:
            return func, grad, hess
        raise NotImplementedError(f"Cannot compute for deriv={deriv}")

    def solve_weights(self, points1, weights0, rcond=1e-14):
        """Assign quadrature weights to exactly match the constraints for the given points.

        Parameters
        ----------
        points1
            Fixed (and optimized) grid point positions.
            It is assumed that the symmetry has already been applied to these points.
        weights0
            The desired weights for the quadrature,
            which are used as a regularization to assign the weights.

        Returns
        -------
        weights1
            The assigned optimal quadrature weights.
        """
        coeffs = [np.ones(len(points1))]
        coeffs.extend(func(points1) for func in self.funcs)
        rhs = np.zeros(len(coeffs))
        rhs[0] = 1
        rhs[1:] = self.targets
        u, s, vh = np.linalg.svd(np.array(coeffs) * weights0, full_matrices=False)
        mask = s > rcond * s[0]
        return np.dot(vh[mask].T, np.dot(u[:, mask].T, rhs) / s[mask]) * weights0


def solve_modified_lm(
    x: NDArray[float],
    equations: Equations,
    rmsdtol: float = 1e-14,
    maxiter: int = 1000,
    maxridge: int = 100,
    verbose: bool = False,
) -> NDArray[float]:
    """Solve the equations using a modified Levenberg-Marquardt algorithm.

    Parameters
    ----------
    x
        The initial guess of the solution.
    equations
        The equations to solve, which must be an instance of the `Equations` class defined above.
    rmsdtol
        The convergence threshold for the RMSD of the equations being solved iteratively.
    maxiter
        The maximum number of iterations for the optimization.
    maxridge
        The maximum number of ridge adjustments for each iteration of the optimization.
    verbose
        If True, print the optimization progress at each iteration.

    Returns
    -------
    x
        The optimized solution.
    """
    if verbose:
        print("iter   RMSD error    Step size        Ridge    Cond.Num.")
        print("----  -----------  -----------  -----------  -----------")
    resids, jac = equations(x, deriv=1)
    rmsd = np.linalg.norm(resids) / np.sqrt(len(resids))
    for i in range(maxiter):
        u, s, vt = np.linalg.svd(jac, full_matrices=False)
        ridge = 0
        for _ in range(maxridge):
            sinv = s / (s**2 + ridge**2)
            dx = -np.dot(vt.T, np.dot(u.T, resids) * sinv)
            xn = x + dx
            if equations.symmetry != Symmetry.NONE:
                xn = abs(xn)
            xn.sort()
            (resids_n,) = equations(xn)
            rmsd_n = np.linalg.norm(resids_n) / np.sqrt(len(resids_n))
            if rmsd_n < rmsd:
                break
            if ridge == 0:
                ridge = s[-1]
            else:
                ridge *= 2
        else:
            raise RuntimeError("step scaling failed")
        x = xn
        resids, jac = equations(x, deriv=1)
        rmsd = rmsd_n
        if verbose:
            print(
                f"{i:4d} {rmsd:12.2e} {np.linalg.norm(dx):12.2e} {ridge:12.2e} {s[0] / s[-1]:12.2e}"
            )
        if rmsd < rmsdtol:
            return x
    raise RuntimeError("solve failed")


def plot_dq(
    extra: dict[str], plot_dist: Callable | None = None, figsize: tuple[int, int] | None = None
):
    """Plot the results of the DQ grid optimization."""
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
        for i, func in enumerate(extra["funcs1"]):
            ax.plot(extra["points1"], func, "+-", label=f"func {i}")
        ax.set_xlabel("x")
        ax.set_ylabel("basis functions")

    plot_points(axs[-2])
    plot_eqs(axs[-1])
    return fig, axs


def dq3(mean: float, std: float, skew: float) -> NDArray:
    r"""Construct a 3-point quadrature grid to integrate degree-3 polynomials exactly.

    The results are exact for an integral of a degree-3 polynomial times
    a distribution with the given mean, standard deviation and skewness.

    Parameters
    ----------
    mean
        The mean of the distribution.
    std
        The standard deviation of the distribution.
    skew
        The skewness of the distribution.
        This value must be in the range $[-1/\sqrt{2}, 1/\sqrt{2}]$,
        otherwise the quadrature points become complex.
        A ValueError is raised if the skewness is out of this range.

    Returns
    -------
    grid
        The 3-point quadrature grid.
    """
    if abs(skew) > 1 / np.sqrt(2):
        raise ValueError("skew must be in the range [-1/sqrt(2), 1/sqrt(2)]")
    grid = np.roots([1, 0, -1.5, -skew]).real
    grid.sort()
    return grid * std + mean
