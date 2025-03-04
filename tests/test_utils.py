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
"""Tests for ``stacie.utils``."""

from numpy.testing import assert_equal

from stacie.utils import block_average, split


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
