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
"""Unit tests for ``stacie.spectrum``."""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from stacie.spectrum import prepare_acfint


def test_basics():
    sequences = np.array(
        [
            [0.66134257, 1.69596962, 2.08533685, 0.62396761, -0.21445517, 1.2226847],
            [-0.66384362, -0.55499254, -1.84284631, 0.3352769, 0.86237774, 0.1605811],
        ]
    )
    prefactor = 0.34
    timestep = 10.0
    spectrum = prepare_acfint(sequences, prefactor=prefactor, timestep=timestep)
    # Test simple properties.
    assert spectrum.nfreq == 4
    assert_equal(spectrum.ndofs, [2, 4, 4, 2])
    assert spectrum.nstep == 6
    assert spectrum.prefactor == prefactor
    assert spectrum.timestep == timestep
    assert spectrum.times[0] == 0.0
    assert spectrum.times[1] == timestep
    assert spectrum.times[-1] == timestep * 5
    assert len(spectrum.times) == 6
    assert spectrum.freqs[0] == 0.0
    assert len(spectrum.freqs) == 4
    assert_allclose(spectrum.freqs[1], 1 / (6 * timestep))
    assert spectrum.amplitudes_ref is None
    # Test the DC-component.
    scale = prefactor * timestep / sequences.shape[1]
    dccomp = (sequences.sum(axis=1) ** 2).mean()
    assert_allclose(spectrum.amplitudes[0], dccomp * scale)
    # Test the Plancherel theorem (taking into account RFFT conventions).
    sumsq = (sequences**2).sum()
    assert_allclose((spectrum.amplitudes * spectrum.ndofs).sum(), sumsq * prefactor * timestep)
    # Test removing the zero frequency
    spectrum2 = spectrum.without_zero_freq()
    assert_equal(spectrum2.ndofs, spectrum.ndofs[1:])
    assert_equal(spectrum2.freqs, spectrum.freqs[1:])
    assert_equal(spectrum2.amplitudes, spectrum.amplitudes[1:])
    assert spectrum2.amplitudes_ref is None
