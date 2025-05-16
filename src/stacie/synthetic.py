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
"""Generate synthetic time-correlated data for algorithmic testing and validation."""

import numpy as np
from numpy.typing import NDArray


def generate(
    psd: NDArray[float],
    timestep: float,
    nseq: int,
    nstep: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[float]:
    """Generate sequences with a given power spectral density.

    Parameters
    ----------
    psd
        The power spectral density.
        The normalization of the PSD is consistent ``compute_spectrum`` when
        using ``prefactors=2.0`` and the given ``timestep`` as arguments.
        The empirical amplitudes of the spectrum will then be consistent with given PSD.
        Hence ``psd[0]`` is the ground truth of the autocorrelation integral.
    timestep
        The time between two subsequent elements in the sequence.
    nseq
        The number of sequences to generate.
    nstep
        The number of time steps in each sequence.
        When not given, the number of steps is ``2 * (len(psd) - 1)``.
        This argument can be used to truncate the sequences,
        which can be useful for creating aperiodic signals.
    rng
        The random number generator.

    Returns
    -------
    sequences
        The generated sequences, a 2D array with shape ``(nseq, nstep)``,
        where ``nstep = 2 * (len(psd) - 1)`` if not provided.
    """
    # Set default arguments
    nstep_max = 2 * (len(psd) - 1)
    if nstep is None:
        nstep = nstep_max
    elif nstep > nstep_max:
        raise ValueError(f"nstep must be at most {nstep_max}")
    if rng is None:
        rng = np.random.default_rng()

    # Generate sequences in the frequency domain and transform to the time domain.
    nfreq = len(psd)
    ft = rng.normal(0, 1, (nseq, nfreq)) + 1.0j * rng.normal(0, 1, (nseq, nfreq))
    ft[:, 0].imag = 0
    ft[:, -1].imag = 0
    ft[:, 1:-1] /= np.sqrt(2)
    ft *= np.sqrt(psd)
    return np.fft.irfft(ft)[:, :nstep] * np.sqrt(nstep_max / timestep)
