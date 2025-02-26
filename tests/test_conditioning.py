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
"""Tests for ``stacie.conditioning``."""

import numdifftools as nd
import numpy as np
from numpy.testing import assert_allclose
from stacie.conditioning import ConditionedCost


def function(x, deriv: int = 0):
    """Compute the product of all items in x, and its gradient and Hessian.

    Parameters
    ----------
    x
        The input vector.
    deriv
        The order of the derivative to compute. Default is 0.
    This is just a simple function to test conditioning implementation.
    """
    results = [float(np.prod(x))]
    if deriv >= 1:
        results.append(results[0] / x)
    if deriv >= 2:
        hess = np.outer(results[1], 1 / x)
        np.fill_diagonal(hess, 0)
        results.append(hess)
    return results


def test_function_deriv1():
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    assert_allclose(
        function(x0, 1)[1],
        nd.Gradient(lambda x: function(x)[0])(x0),
        atol=1e-12,
        rtol=1e-12,
    )


def test_function_deriv2():
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    assert_allclose(
        function(x0, 2)[2],
        nd.Gradient(lambda x: function(x, deriv=1)[1])(x0),
        atol=1e-12,
        rtol=1e-12,
    )


def test_conditioned_cost():
    par_scales = np.array([1.0, 2.0, 3.0, 4.0])
    cost = ConditionedCost(function, par_scales, 5.0)
    x0 = np.array([0.1, 0.2, 0.3, 0.4])
    assert_allclose(cost(x0, 0), [np.prod(x0 * par_scales) / 5.0])
    assert_allclose(cost.from_reduced(x0), x0 * par_scales)
    assert_allclose(cost.to_reduced(x0), x0 / par_scales)
    assert_allclose(cost.funcgrad(x0)[0], cost(x0)[0])
    assert_allclose(cost.funcgrad(x0)[1], cost(x0, 1)[1])
    assert_allclose(cost.hess(x0), cost(x0, 2)[2])


def test_conditioned_cost_deriv1():
    par_scales = np.array([1.0, 2.0, 3.0, 4.0])
    cost = ConditionedCost(function, par_scales, 5.0)
    x0 = np.array([0.1, 0.2, 0.3, 0.4])
    assert_allclose(
        cost(x0, 1)[1],
        nd.Gradient(lambda x: cost(x)[0])(x0),
        atol=1e-12,
        rtol=1e-12,
    )


def test_conditioned_cost_deriv2():
    par_scales = np.array([1.0, 2.0, 3.0, 4.0])
    cost = ConditionedCost(function, par_scales, 5.0)
    x0 = np.array([0.1, 0.2, 0.3, 0.4])
    assert_allclose(
        cost(x0, 2)[2],
        nd.Gradient(lambda x: cost(x, deriv=1)[1])(x0),
        atol=1e-12,
        rtol=1e-12,
    )
