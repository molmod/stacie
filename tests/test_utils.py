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
"""Tests for ``stacie.utils``."""

import numpy as np
import pytest
from numpy.testing import assert_equal

from stacie.utils import block_average, mixture_stats, robust_dot, robust_posinv, split


def test_split_sequences():
    assert_equal(split([1, 2, 3, 4, 5, 6], 2), [[1, 2, 3], [4, 5, 6]])
    assert_equal(split([1, 2, 3, 4, 5, 6, 7], 2), [[1, 2, 3], [4, 5, 6]])
    assert_equal(split([[1, 2, 3, 4, 5, 6]], 2), [[1, 2, 3], [4, 5, 6]])
    assert_equal(split([[1, 2, 3, 4, 5, 6, 7]], 2), [[1, 2, 3], [4, 5, 6]])
    assert_equal(split([[1, 2, 3, 4], [5, 6, 7, 8]], 2), [[1, 2], [3, 4], [5, 6], [7, 8]])
    assert_equal(split([[1, 2, 3, 4, -1], [5, 6, 7, 8, -2]], 2), [[1, 2], [3, 4], [5, 6], [7, 8]])


def test_block_average():
    assert_equal(block_average([1, 2, 3, 4, 5, 6], 2), [[1.5, 3.5, 5.5]])
    assert_equal(block_average([1, 2, 3, 4, 5, 6, 7], 2), [[1.5, 3.5, 5.5]])
    assert_equal(block_average([[1, 2, 3, 4, 5, 6]], 2), [[1.5, 3.5, 5.5]])
    assert_equal(block_average([[1, 2, 3, 4, 5, 6, 7]], 2), [[1.5, 3.5, 5.5]])
    assert_equal(block_average([[1, 2, 3, 4], [5, 6, 7, 8]], 3), [[2.0], [6.0]])
    assert_equal(
        block_average(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15, 16],
            ],
            4,
        ),
        [[2.5, 6.5], [10.5, 14.5]],
    )


@pytest.mark.parametrize(
    "a",
    [
        [[8, 2], [2, 4]],
        [[7, 2, 3], [2, 6, 1], [3, 1, 9]],
    ],
)
def test_robust_posinv_simple(a):
    scales, evals, evecs, inv = robust_posinv(a)
    assert scales.shape == (len(a),)
    assert evals.shape == (len(a),)
    assert evecs.shape == (len(a), len(a))
    assert inv.shape == (len(a), len(a))
    assert (evals > 0).all()
    assert inv == pytest.approx(inv.T)
    assert np.dot(a, inv) == pytest.approx(np.eye(len(a)), abs=1e-10)


def test_robust_posinv_hard():
    basis = np.array(
        [
            [10, 1e15, -3e-15],
            [2, 1e18, -5e-14],
            [3.7, -5e17, 1e-18],
        ]
    )
    a = np.dot(basis.T, basis)
    scales, evals, evecs, inv = robust_posinv(a)
    assert scales.shape == (len(a),)
    assert evals.shape == (len(a),)
    assert evecs.shape == (len(a), len(a))
    assert inv.shape == (len(a), len(a))
    assert (evals > 0).all()
    print(inv)
    assert inv == pytest.approx(inv.T)
    half = np.dot(basis, inv)
    assert np.dot(half.T, half) == pytest.approx(inv, rel=1e-10)


@pytest.mark.parametrize("shape", [(3,), (3, 5)])
def test_robust_dot_simple(shape):
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((3, 3))
    a = np.dot(basis.T, basis)
    scales, evals, evecs, inv = robust_posinv(a)
    other = rng.standard_normal(shape)
    result = robust_dot(scales, evals, evecs, other)
    assert result.shape == shape
    assert result == pytest.approx(np.dot(a, other))
    result = robust_dot(1 / scales, 1 / evals, evecs, other)
    assert result.shape == shape
    assert result == pytest.approx(np.dot(inv, other))


def test_mixture_stats():
    means = np.array([[1, 2], [3, 4]])
    covars = np.array([[[1.2, 2.5], [3.1, 1.1]], [[7.3, 8.1], [2.2, 3.1]]])
    weights = np.array([2.0, 8.0])
    mean1, covar1 = mixture_stats(means, covars, weights)
    assert mean1 == pytest.approx([2.6, 3.6])
    assert covar1 == pytest.approx(
        0.2 * covars[0]
        + 0.8 * covars[1]
        + 0.2 * np.outer(means[0] - mean1, means[0] - mean1)
        + 0.8 * np.outer(means[1] - mean1, means[1] - mean1)
    )
    mean2, covar2 = mixture_stats(means, np.einsum("ijj->ij", covars), weights)
    assert mean2 == pytest.approx(mean1)
    assert covar2 == pytest.approx(np.diag(covar1))
