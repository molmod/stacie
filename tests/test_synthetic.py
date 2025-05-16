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
"""Unit tests for ``stacie.synthetic``"""

import numpy as np
import pytest

from stacie.spectrum import compute_spectrum
from stacie.synthetic import generate


def test_shape_none():
    psd = np.array([1, 2, 3, 4, 5])
    nseq = 3
    sequences = generate(psd, 1.0, nseq)
    assert sequences.shape == (nseq, 8)


def test_shape_shorter():
    psd = np.array([1, 2, 3, 4, 5])
    nseq = 3
    sequences = generate(psd, 1.0, nseq, 6)
    assert sequences.shape == (nseq, 6)


def test_shape_longer():
    psd = np.array([1, 2, 3, 4, 5])
    nseq = 3
    with pytest.raises(ValueError):
        generate(psd, 1.0, nseq, 20)


@pytest.mark.parametrize("truncate", [False, True])
def test_psd(truncate: bool):
    # Set up a test case with a Lorentzian power spectral density.
    tau = 5.0
    h = 0.1
    nstep = 4096
    tmax = h * nstep
    nfreq = nstep // 2 + 1
    freq0 = 1 / tmax
    omegas = np.arange(nfreq) * 2 * np.pi * freq0
    psd = tau / (1 + (omegas * tau) ** 2)
    nseq = 1024
    if truncate:
        nstep = 2048
    rng = np.random.default_rng(42)
    sequences = generate(psd, h, nseq, nstep, rng)

    # Test the shape of the output.
    assert sequences.shape == (nseq, nstep)
    # Test the consistency with compute_spectrum.
    spectrum = compute_spectrum(sequences, prefactors=2.0, timestep=h)
    if truncate:
        # The spectrum subsampled due to the truncation.
        psd = psd[::2]
    assert spectrum.amplitudes[:512] == pytest.approx(psd[:512], rel=0.2)
    # Test the Plancherel theorem. Origins of the factors 2:
    # - One factor two from the RFFT. (The PSD is one-sided.)
    # - If truncated, there is a second factor 2, because the power spectrum
    #   in compute_spectrum has a factor 1/N, and the generate function
    #   is compatible with this convention by including factor 1/sqrt(N) in the inverse RFFT.
    true_variance = 2 * psd.sum() * freq0
    if truncate:
        true_variance *= 2
    assert (sequences**2).mean() == pytest.approx(true_variance, rel=0.1)
