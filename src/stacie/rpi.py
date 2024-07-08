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
"""Algorithms for tuning a positive integer hyperparameter.

The shorthand "rpi" stands for "robust positive integer".
"""

import bisect

import numpy as np

__all__ = ("rpi_opt",)


def rpi_opt(func, xbracket, *, nsplit: int = 5, nsweep=20, budget=20, cache=None, mode="max"):
    """Robust positive integer generalization of bisection optimization.

    Parameters
    ----------
    func
        The function (of an integer) to be maxi- or minimized to approximation.
    xbracket
        The initial bracket to consider.
        This must be a sorted tuple of at least two potential arguments.
        for func, but more may be added, in which case they are included
        in the grid search.
        Extra points may be useful when they are known to be important.
    nsplit
        The number of splits of the bracket after the initial recursion.
        These will be linearly distributed.
    nsweep
        The (approximate) number of grid points in the initial scan.
        These will be uniformly distributed on a log scale.
    budget
        The number of times the algorithm will try converging an alternative
        solutions when these appear to be present.
    cache
        A dictionary with cache ``(x, func(x))`` items.
        When this is empty or ``None``, the initial scan with nsweep points,
        is performed first.
        When the function returns, this dictionary will contain all function
        evaluations.
    mode
        The type of extremum: "min" or "max".

    Returns
    -------
    budget
        The remaining budget.
    """
    if not (budget >= 0):
        raise ValueError("Requirement not met: budget >= 0")
    if mode not in ["min", "max"]:
        raise ValueError('Requirement not met: mode in ["min", "max"]')

    # Decide what to do
    sign = -1 if mode == "max" else +1

    todo = [(0, xbracket)]

    while len(todo) > 0 and budget > 0:
        fscore, xbracket = todo.pop(0)
        if not np.isfinite(fscore):
            continue

        xgrid, fgrid, cache = build_eval_grid(func, xbracket, nsplit, nsweep, cache)
        useful = False
        budget -= 1

        # Regular sub brackets for local extrema.
        cases = zip(
            fgrid[:-2], fgrid[1:-1], fgrid[2:], xgrid[:-2], xgrid[1:-1], xgrid[2:], strict=True
        )
        for f0, f1, f2, x0, x1, x2 in cases:
            if sign * f1 <= min(sign * f0, sign * f2) and (x2 - x0) > 2:
                bisect.insort(todo, (sign * f1, (x0, x1, x2)))
                useful = True
        # Also explore edge cases
        if sign * fgrid[0] < sign * fgrid[1] and xgrid[1] - xgrid[0] > 1:
            bisect.insort(todo, (sign * fgrid[0], (xgrid[0], xgrid[1])))
            useful = True
        if sign * fgrid[-1] < sign * fgrid[-2] and xgrid[-1] - xgrid[-2] > 1:
            bisect.insort(todo, (sign * fgrid[-1], (xgrid[-2], xgrid[-1])))
            useful = True

        if useful:
            budget += 1

    return budget


def build_eval_grid(func, xbracket, nsplit=5, nsweep=100, cache=None):
    """Build and evaluate the grid.

    See rpi_opt docstring for details.

    Returns
    -------
    func
        The function to be evaluated on the grid.
    xgrid
        The grid of integer arguments.
        It is guaranteed that the first and last one correspond to the bracket.
    fgrid
        The corresponding function values.
    cache
        The new or given (and updated) cache object.
    """
    if not (xbracket[0] >= 0):
        raise ValueError("Requirement not met: xbracket[0] >= 0")
    if not (xbracket[-1] >= xbracket[0]):
        raise ValueError("Requirement not met: bracket[-1] >= bracket[0]")
    for i in range(len(xbracket) - 1):
        if not (xbracket[i + 1] > xbracket[i]):
            raise ValueError(f"Requirement not met: xbracket[{i + 1}] > xbracket[{i}]")
    if not (nsplit >= 2):
        raise ValueError("Requirement not met: nsplit >= 2")
    if not (nsweep >= 2):
        raise ValueError("Requirement not met: nsweep >= 2")

    cache = {} if cache is None else cache
    if len(cache) == 0:
        xgrid = _build_xgrid_exp(xbracket, nsweep)
    else:
        xgrid = _build_xgrid_lin(xbracket, nsplit)
    assert xgrid[0] == xbracket[0]
    assert xgrid[-1] == xbracket[-1]
    fgrid = _eval_grid(xgrid, func, cache)
    return xgrid, fgrid, cache


def _build_xgrid_lin(xbracket, nsplit):
    """Approximate linear integer grid."""
    step = max(1, (xbracket[-1] - xbracket[0]) / nsplit)
    if step == 1:
        return list(range(xbracket[0], xbracket[-1] + 1))
    xgrid = {int(round(xbracket[0] + step * isplit)) for isplit in range(nsplit + 1)}
    xgrid.update(xbracket)
    return sorted(xgrid)


def _build_xgrid_exp(xbracket, ngrid):
    """Approximate exponential integer grid."""
    xgrid = [xbracket[0]]
    while len(xgrid) < ngrid:
        # Update the exponent as we go.
        # At lower values, integer spacing will be wider than the exponential
        # grid, which is compensated for by decreasing the exponent.
        logratio = np.log(xbracket[-1]) - np.log(max(1, xgrid[-1]))
        alpha = np.exp(logratio / (ngrid - len(xgrid)))
        assert alpha >= 1
        newx = max(xgrid[-1] + 1, int(np.ceil(alpha * xgrid[-1])))
        if newx > xbracket[-1]:
            break
        xgrid.append(newx)
    xgrid = set(xgrid)
    xgrid.update(xbracket)
    return sorted(xgrid)


def _eval_grid(xgrid, func, cache):
    fgrid = []
    for x in xgrid:
        f = cache.get(x)
        if f is None:
            f = func(x)
            cache[x] = f
        fgrid.append(f)
    return fgrid
