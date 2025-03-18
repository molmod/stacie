# Stacie is a STable AutoCorrelation Integral Estimator.
# Copyright (C) 2024-2025 The contributors of the Stacie Python Package.
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
"""Unit tests for ``stacie.cutoff``."""

import numpy as np
import pytest

from stacie.cutoff import expected_ufc, general_ufc


def test_ufc_expectation_value():
    rng = np.random.default_rng(42)
    nfreq = 30
    basis = rng.standard_normal((4, nfreq))
    exp = expected_ufc(basis)

    ufcs = []
    for _ in range(10000):
        data = rng.standard_normal(nfreq)
        data -= np.linalg.lstsq(basis.T, data, rcond=None)[0] @ basis
        ufc = general_ufc(data)
        ufcs.append(ufc)
    assert exp == pytest.approx(np.mean(ufcs), rel=1e-1)
