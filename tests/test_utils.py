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
from stacie.utils import split_sequences


def test_split_sequences():
    assert_equal(split_sequences([1, 2, 3, 4, 5, 6], 2), [[1, 2, 3], [4, 5, 6]])
    assert_equal(split_sequences([1, 2, 3, 4, 5, 6, 7], 2), [[1, 2, 3], [4, 5, 6]])
    assert_equal(split_sequences([[1, 2, 3, 4, 5, 6]], 2), [[1, 2, 3], [4, 5, 6]])
    assert_equal(split_sequences([[1, 2, 3, 4, 5, 6, 7]], 2), [[1, 2, 3], [4, 5, 6]])
    assert_equal(split_sequences([[1, 2, 3, 4], [5, 6, 7, 8]], 2), [[1, 2], [3, 4], [5, 6], [7, 8]])
    assert_equal(
        split_sequences([[1, 2, 3, 4, -1], [5, 6, 7, 8, -2]], 2), [[1, 2], [3, 4], [5, 6], [7, 8]]
    )
