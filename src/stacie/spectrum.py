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
"""Utility to prepare the spectrum and other inputs for given sequences."""

from typing import Self

import attrs
import numpy as np
from numpy.typing import NDArray

from .utils import split

__all__ = ("Spectrum", "compute_spectrum")


@attrs.define
class Spectrum:
    """Container class holding all the inputs for the autocorrelation integral estimate."""

    ndofs: NDArray[int] = attrs.field()
    """The number of independnt contributions to each amplitude."""

    prefactor: float = attrs.field()
    """The given prefactor for the spectrum to fix the units for the autocorrelation integral."""

    variance: float = attrs.field()
    """The variance of the input sequences."""

    timestep: float = attrs.field()
    """The time between two subsequent elements in the given sequence."""

    nstep: int = attrs.field()
    """The number of time steps in the input."""

    freqs: NDArray[float] = attrs.field()
    """The equidistant frequency axis of the spectrum."""

    amplitudes: NDArray[float] = attrs.field()
    """The spectrum amplitudes averaged over the given input sequences."""

    amplitudes_ref: NDArray[float] | None = attrs.field(default=None)
    """Optionally, the known analytical model of the power spectrum, on the same frequency grid."""

    @property
    def nfreq(self) -> int:
        """The number of RFFT frequency grid points."""
        return len(self.freqs)

    def without_zero_freq(self) -> Self:
        """Return a copy without the DC component."""
        return attrs.evolve(
            self,
            ndofs=self.ndofs[1:],
            freqs=self.freqs[1:],
            amplitudes=self.amplitudes[1:],
            amplitudes_ref=None if self.amplitudes_ref is None else self.amplitudes_ref[1:],
        )


def compute_spectrum(
    sequences: NDArray[float],
    *,
    prefactor: float = 0.5,
    timestep: float = 1,
    include_zero_freq: bool = True,
    nsplit: int = 1,
) -> Spectrum:
    """Compute a spectrum and store all inputs for ``estimate_acint`` in a ``Spectrum`` instance.

    Parameters
    ----------
    sequences
        The input sequences, array with shape ``(nindep, nstep)``,
        of which each row is a time-dependent sequence.
        All sequences are assumed to be statistically independent and have length ``nstep``.
        (Time correlations within one sequence are fine, obviously.)
        You may also provide a single sequence,
        in which case the shape of the array is ``(nstep,)``.
        However, we recommend using multiple independent sequences to reduce uncertainties.
    prefactor
        A factor to be multiplied with the autocorrelation function
        to give it a physically meaningful unit.
    timestep
        The time step of the input sequence.
    include_zero_freq
        When set to False, the DC component of the spectrum is discarded.
    nsplit
        If larger than 1, the sequences are split into ``nsplit`` chuncks
        and the spectrum of each chunck is computed separately, as if they are separate sequences.
        This reduces the resolution of the frequency axis and the variance of the spectrum.
        The function ``stacie.utils.split`` is used for this purpose.

    Returns
    -------
    spectrum
        A ``Spectrum`` object holding all the inputs needed to estimate
        the integral of the autocorrelation function.
    """
    # Handle single sequence case
    if sequences.ndim == 1:
        sequences = sequences.reshape(1, -1)
    elif sequences.ndim != 2:
        raise ValueError("Sequences must be a 1D or 2D array.")

    # Split the sequences into chunks for better statistics if requested.
    if nsplit > 1:
        sequences = split(sequences, nsplit)

    # Get basic parameters of the input sequences.
    nindep, nstep = sequences.shape
    freqs = np.fft.rfftfreq(nstep, d=timestep)

    # Number of "degrees of freedom" (contributions) to each amplitude
    ndofs = np.full(freqs.shape, 2 * nindep)
    ndofs[0] = nindep
    if len(freqs) % 2 == 0:
        ndofs[-1] = nindep

    # Compute the spectrum and scale it.
    amplitudes = (abs(np.fft.rfft(sequences, axis=1)) ** 2).mean(axis=0)
    amplitudes *= prefactor * timestep / nstep

    # Remove DC component, useful for inputs that oscillate about a non-zero average.
    if not include_zero_freq:
        ndofs = ndofs[1:]
        freqs = freqs[1:]
        amplitudes = amplitudes[1:]

    return Spectrum(ndofs, prefactor, sequences.var(), timestep, nstep, freqs, amplitudes)
