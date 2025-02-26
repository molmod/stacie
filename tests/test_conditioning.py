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

import numpy as np
import pytest
from conftest import check_gradient, check_hessian
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
    check_gradient(function, x0)


def test_function_deriv2():
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    check_hessian(function, x0)


def test_conditioned_cost():
    par_scales = np.array([1.0, 2.0, 3.0, 4.0])
    cost = ConditionedCost(function, par_scales, 5.0)
    x0 = np.array([0.1, 0.2, 0.3, 0.4])
    assert cost(x0, 0) == pytest.approx([np.prod(x0 * par_scales) / 5.0])
    assert cost.from_reduced(x0) == pytest.approx(x0 * par_scales)
    assert cost.to_reduced(x0) == pytest.approx(x0 / par_scales)
    assert cost.funcgrad(x0)[0] == pytest.approx(cost(x0)[0])
    assert cost.funcgrad(x0)[1] == pytest.approx(cost(x0, 1)[1])
    assert cost.hess(x0) == pytest.approx(cost(x0, 2)[2])


def test_conditioned_cost_deriv1():
    par_scales = np.array([1.0, 2.0, 3.0, 4.0])
    cost = ConditionedCost(function, par_scales, 5.0)
    x0 = np.array([0.1, 0.2, 0.3, 0.4])
    check_gradient(cost, x0)


def test_conditioned_cost_deriv2():
    par_scales = np.array([1.0, 2.0, 3.0, 4.0])
    cost = ConditionedCost(function, par_scales, 5.0)
    x0 = np.array([0.1, 0.2, 0.3, 0.4])
    check_hessian(cost, x0)
