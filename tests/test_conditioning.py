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
        For vectorized calculations, use N-dimensional inputs of which the last axis
        corresponds to the parameter index.
    deriv
        The order of the derivative to compute. Default is 0.
    This is just a simple function to test conditioning implementation.
    """
    results = [np.prod(x, axis=-1)]
    if deriv >= 1:
        results.append(np.einsum("...,...i->...i", results[0], 1 / x))
    if deriv >= 2:
        hess = np.einsum("...i,...j->...ij", results[1], 1 / x)
        for i in range(x.shape[-1]):
            hess[..., i, i] = 0
        results.append(hess)
    return results


def test_vectorize_function():
    """Check that the function can be vectorized."""
    x0 = np.array([[1.0, 2.0, 4.0], [1.0, 3.0, 4.0]])
    results = function(x0, deriv=2)
    assert results[0].shape == (2,)
    assert results[1].shape == (2, 3)
    assert results[2].shape == (2, 3, 3)
    for i, one_x0 in enumerate(x0):
        one_results = function(one_x0, deriv=2)
        assert one_results[0].shape == ()
        assert one_results[1].shape == (3,)
        assert one_results[2].shape == (3, 3)
        assert np.all(results[0][i] == one_results[0])
        assert np.all(results[1][i] == one_results[1])
        assert np.all(results[2][i] == one_results[2])


def test_function_deriv1():
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    check_gradient(function, x0)


def test_function_deriv2():
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    check_hessian(function, x0)


def test_vectorize_conditioned_cost():
    """Check that the conditioned cost can be vectorized."""
    par_scales = np.array([5.0, 2.3, 7.0])
    cost = ConditionedCost(function, par_scales, 5.0)
    x0 = np.array([[1.0, 2.0, 3.0], [1.0, 3.0, 4.0]])
    results = cost(x0, deriv=2)
    assert results[0].shape == (2,)
    assert results[1].shape == (2, 3)
    assert results[2].shape == (2, 3, 3)
    for i, one_x0 in enumerate(x0):
        one_results = cost(one_x0, deriv=2)
        assert one_results[0].shape == ()
        assert one_results[1].shape == (3,)
        assert one_results[2].shape == (3, 3)
        assert np.all(results[0][i] == one_results[0])
        assert np.all(results[1][i] == one_results[1])
        assert np.all(results[2][i] == one_results[2])


def test_conditioned_cost():
    par_scales = np.array([1.0, 2.0, 3.0, 4.0])
    cost = ConditionedCost(function, par_scales, 5.0)
    x0 = np.array([0.1, 0.2, 0.3, 0.4])
    assert cost(x0, deriv=0) == pytest.approx([np.prod(x0 * par_scales) / 5.0])
    assert cost.from_reduced(x0) == pytest.approx(x0 * par_scales)
    assert cost.to_reduced(x0) == pytest.approx(x0 / par_scales)
    assert cost.funcgrad(x0)[0] == pytest.approx(cost(x0)[0])
    assert cost.funcgrad(x0)[1] == pytest.approx(cost(x0, deriv=1)[1])
    assert cost.hess(x0) == pytest.approx(cost(x0, deriv=2)[2])


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
