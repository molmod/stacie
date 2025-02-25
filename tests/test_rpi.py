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
"""Unit tests for ``acint.rpi``."""

import numpy as np
import pytest
from stacie.rpi import rpi_opt


def test_rpi_opt_simple_min():
    def cost(x):
        return np.exp(-x + 5) + x

    cache = {}
    rpi_opt(cost, (0, 30), cache=cache, mode="min")
    assert cache[5] == pytest.approx(6.0)


def test_rpi_opt_simple_max():
    def objective(x):
        return x - np.exp(x - 5)

    cache = {}
    rpi_opt(objective, (0, 30), cache=cache, mode="max")
    assert cache[5] == pytest.approx(4.0)


def test_rpi_opt_wavy_min():
    def cost(x):
        return -np.cos(2 * np.pi * x) + (np.exp(-x + 3) + x)

    cache = {}
    rpi_opt(cost, (0, 30), cache=cache, mode="min")
    assert cache[3] == pytest.approx(3.0)


def test_rpi_opt_wavy_max():
    def objective(x):
        return np.cos(2 * np.pi * (0.1 * x) ** 2)

    cache = {}
    rpi_opt(objective, (0, 20), cache=cache, mode="max")
    assert cache[0] == pytest.approx(1.0)
    assert cache[10] == pytest.approx(1.0)


def test_rpi_opt_edge():
    def objective(x):
        return np.exp(x / 10)

    cache = {}
    rpi_opt(objective, (0, 10), cache=cache, mode="max", nsweep=2)
    assert 9 in cache
