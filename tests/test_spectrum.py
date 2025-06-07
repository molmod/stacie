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
"""Unit tests for ``stacie.spectrum``."""

import numpy as np
import pytest
from numpy.testing import assert_equal

from stacie.spectrum import compute_spectrum


@pytest.mark.parametrize("use_iter", [False, True])
@pytest.mark.parametrize("prefactors_none", [False, True])
def test_basics(use_iter: bool, prefactors_none: bool):
    sequences = np.array(
        [
            [0.66134257, 1.69596962, 2.08533685, 0.62396761, -0.21445517, 1.2226847],
            [-0.66384362, -0.55499254, -1.84284631, 0.3352769, 0.86237774, 0.1605811],
        ]
    )
    prefactor = 0.34
    timestep = 10.0
    if prefactors_none:
        spectrum = compute_spectrum(
            [(prefactor, sequences[0]), (prefactor, sequences[1])]
            if use_iter
            else (prefactor, sequences),
            prefactors=None,
            timestep=timestep,
        )
    else:
        spectrum = compute_spectrum(
            iter(sequences) if use_iter else sequences,
            prefactors=[prefactor] * 2 if use_iter else prefactor,
            timestep=timestep,
        )
    # Test simple properties.
    assert spectrum.nfreq == 4
    assert_equal(spectrum.ndofs, [2, 4, 4, 2])
    assert spectrum.nstep == 6
    assert spectrum.timestep == timestep
    assert spectrum.freqs[0] == 0.0
    assert len(spectrum.freqs) == 4
    assert spectrum.freqs[1] == pytest.approx(1 / (6 * timestep))
    assert spectrum.amplitudes_ref is None
    # Test the DC-component.
    scale = 0.5 * prefactor * timestep / sequences.shape[1]
    dccomp = (sequences.sum(axis=1) ** 2).mean()
    assert spectrum.amplitudes[0] == pytest.approx(dccomp * scale)
    # Test the Plancherel theorem (taking into account RFFT conventions).
    sumsq = (sequences**2).sum()
    assert (spectrum.amplitudes * spectrum.ndofs).sum() == pytest.approx(
        sumsq * 0.5 * prefactor * timestep
    )
    # Test removing the zero frequency
    spectrum2 = spectrum.without_zero_freq()
    assert_equal(spectrum2.ndofs, spectrum.ndofs[1:])
    assert_equal(spectrum2.freqs, spectrum.freqs[1:])
    assert_equal(spectrum2.amplitudes, spectrum.amplitudes[1:])
    assert spectrum2.amplitudes_ref is None


def test_single():
    sequence = np.array([0.66134257, 1.69596962, 2.08533685, 0.62396761, -0.21445517, 1.2226847])
    prefactor = 0.25
    timestep = 2.5
    spectrum = compute_spectrum(sequence, prefactors=prefactor, timestep=timestep)
    # Test simple properties.
    assert spectrum.nfreq == 4
    assert_equal(spectrum.ndofs, [1, 2, 2, 1])
    assert spectrum.nstep == 6
    assert spectrum.timestep == timestep
    assert spectrum.freqs[0] == 0.0
    assert len(spectrum.freqs) == 4
    assert spectrum.freqs[1] == pytest.approx(1 / (6 * timestep))
    assert spectrum.amplitudes_ref is None
    # Test the DC-component.
    scale = 0.5 * prefactor * timestep / sequence.shape[0]
    dccomp = sequence.sum() ** 2
    assert spectrum.amplitudes[0] == pytest.approx(dccomp * scale)
    # Test the Plancherel theorem (taking into account RFFT conventions).
    sumsq = (sequence**2).sum()
    assert (spectrum.amplitudes * spectrum.ndofs).sum() == pytest.approx(
        sumsq * 0.5 * prefactor * timestep
    )
    # Test removing the zero frequency
    spectrum2 = spectrum.without_zero_freq()
    assert_equal(spectrum2.ndofs, spectrum.ndofs[1:])
    assert_equal(spectrum2.freqs, spectrum.freqs[1:])
    assert_equal(spectrum2.amplitudes, spectrum.amplitudes[1:])
    assert spectrum2.amplitudes_ref is None


def test_mixed():
    sequences = [
        np.array([0.66134257, 1.69596962, 2.08533685, 0.62396761, -0.21445517, 1.2226847]),
        np.array([-0.66384362, -0.55499254, -1.84284631, 0.3352769, 0.86237774, 0.1605811]),
    ]
    prefactors1 = 3.1
    prefactors2 = [3.1, 3.1]
    spectrum1 = compute_spectrum(sequences, prefactors=prefactors1, timestep=2.5)
    spectrum2 = compute_spectrum(sequences, prefactors=prefactors2, timestep=2.5)
    assert spectrum1.amplitudes == pytest.approx(spectrum2.amplitudes)


def test_variance():
    sequences = np.array(
        [
            [0.90969991, 0.09442405, 0.98258277, 0.43112388, 0.81369217, 0.5543517, 0.10178618],
            [0.78626291, 0.9948804, 0.2698581, 0.77724192, 0.11130753, 0.51409317, 0.34561125],
        ]
    )
    print(sequences.mean() ** 2)
    s1 = compute_spectrum(sequences)
    s2 = s1.without_zero_freq()
    s3 = compute_spectrum(sequences, include_zero_freq=False)
    assert s1.variance == pytest.approx((sequences**2).mean())
    assert s2.variance == pytest.approx(np.var(sequences, ddof=1))
    assert s3.variance == pytest.approx(np.var(sequences, ddof=1))
