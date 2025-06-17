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
"""Utility to prepare the spectrum and other inputs for given sequences."""

from collections.abc import Iterable
from itertools import repeat
from typing import Self

import attrs
import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ("Spectrum", "compute_spectrum")


@attrs.define
class Spectrum:
    """Container class holding all the inputs for the autocorrelation integral estimate."""

    mean: float = attrs.field(converter=float)
    """The mean of the input sequences multiplied by the square root of the prefactor."""

    variance: float = attrs.field(converter=float)
    """The variance of the input sequences multiplied by the prefactor."""

    timestep: float = attrs.field(converter=float)
    """The time between two subsequent elements in the given sequence."""

    nstep: int = attrs.field(converter=int)
    """The number of time steps in the input."""

    freqs: NDArray[float] = attrs.field()
    """The equidistant frequency axis of the spectrum."""

    ndofs: NDArray[float] = attrs.field()
    """The number of independent contributions to each amplitude."""

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
        if self.freqs[0] != 0.0:
            raise ValueError("The zero frequency has already been removed.")
        variance = self.variance - self.mean**2
        nindep = self.ndofs[0]
        variance *= (nindep * self.nstep) / (nindep * self.nstep - 1)
        return attrs.evolve(
            self,
            variance=variance,
            freqs=self.freqs[1:],
            ndofs=self.ndofs[1:],
            amplitudes=self.amplitudes[1:],
            amplitudes_ref=None if self.amplitudes_ref is None else self.amplitudes_ref[1:],
        )


def compute_spectrum(
    sequences: Iterable[NDArray[float]] | NDArray[float],
    *,
    prefactors: Iterable[NDArray[float]] | NDArray[float] | None = 1.0,
    timestep: float = 1,
    include_zero_freq: bool = True,
) -> Spectrum:
    r"""Compute a spectrum and store all inputs for ``estimate_acint`` in a ``Spectrum`` instance.

    The spectrum amplitudes are computed as follows:

    .. math::

        C_k = \frac{1}{M}\sum_{m=1}^M \frac{F_m h}{2 N} \left|
            \sum_{n=0}^{N-1} x^{(m)}_n \exp\left(-i \frac{2 \pi n k}{N}\right)
        \right|^2

    where:

    - :math:`F_m` is the given prefactor (may be different for each sequence),
    - :math:`h` is the timestep,
    - :math:`N` is the number of time steps in the input sequences,
    - :math:`M` is the number of independent sequences,
    - :math:`x^{(m)}_n` is the value of the :math:`m`-th sequence at time step :math:`n`,
    - :math:`k` is the frequency index.

    The sum over :math:`m` simply averages spectra obtained from different sequences.
    The factor :math:`F_m h/ 2 N` normalizes the spectrum so that its zero-frequency limit
    is an estimate of the autocorrelation integral.

    Parameters
    ----------
    sequences
        The input sequences, which can have several forms.
        If ``prefactors`` is not ``None``, it can be:

        - An array with shape ``(nindep, nstep)`` or ``(nstep,)``.
          In case of a 2D array, each row is a time-dependent sequence.
          In case of a 1D array, a single sequence is used.
        - An iterable whose items are arrays as described in the previous point.
          This option is convenient when a single array does not fit in memory.

        If ``prefactors`` is ``None``:

        - A tuple of a prefactor (or an array of prefactors) and a sequences array,
          either 1D or 2D as described above.
        - An iterable whose items are tuples of a prefactor (or an array of prefactors)
          and a sequences array, either 1D or 2D as described above.

        All sequences are assumed to be statistically independent and have length ``nstep``.
        (Time correlations within one sequence are fine, obviously.)
        We recommend using multiple independent sequences to reduce uncertainties.
        Arrays must be used. (lists of floating point values are not supported.)
    prefactors
        A positive factor to be multiplied with the autocorrelation function
        to give it a physically meaningful unit.
        This argument can be given in multiple forms:

        - None, in which case the sequences are assumed to be
          one or more (prefactors, sequences) tuples.
        - A single floating point value: the same prefactor is used for all input sequences.
        - A single array with shape ``(nindep,)``:
          each sequence is multiplied with the corresponding prefactor.
        - An iterable whose items are of the form described in the previous two points.
          In this case, the sequences must also be given as an iterable with the same length.
    timestep
        The time step of the input sequence.
    include_zero_freq
        When set to False, the DC component of the spectrum is discarded.

    Returns
    -------
    spectrum
        A ``Spectrum`` object holding all the inputs needed to estimate
        the integral of the autocorrelation function.
    """
    # Handle tuple (prefactor, sequences) case
    if isinstance(sequences, tuple):
        prefactors, sequences = sequences

    # Handle single-array case, assume consistency between sequences and prefactor.
    if isinstance(sequences, np.ndarray):
        sequences = [sequences]
        prefactors = [prefactors]

    # Process iterable of arrays
    if isinstance(sequences, Iterable):
        nindep = 0
        nstep = None
        amplitudes = 0
        total = 0
        total_sq = 0
        if prefactors is None:
            iterator = sequences
        elif isinstance(prefactors, Iterable):
            iterator = zip(prefactors, sequences, strict=True)
        else:
            iterator = zip(repeat(prefactors), sequences, strict=False)
        for item_prefactors, item_sequences in iterator:
            nstep, item_nindep, item_amplitudes, item_total, item_total_sq = _process_sequences(
                item_sequences, item_prefactors, nstep
            )
            nindep += item_nindep
            amplitudes += item_amplitudes
            total += item_total
            total_sq += item_total_sq
    else:
        raise TypeError("The sequence argument must be an array or an iterable of arrays.")

    # Frequency axis and scale of amplitudes
    freqs = np.fft.rfftfreq(nstep, d=timestep)
    amplitudes *= timestep / nindep

    # Number of "degrees of freedom" (contributions) to each amplitude
    ndofs = np.full(freqs.shape, 2 * nindep, dtype=float)
    ndofs[0] = nindep
    if len(freqs) % 2 == 0:
        ndofs[-1] = nindep

    # Remove DC component, useful for inputs that oscillate about a non-zero average.
    # The variance is calculated consistently:
    # - If the DC component is removed, the variance is calculated with respect to the mean.
    # - Otherwise, the variance is calculated with respect to zero.
    mean = total / nindep
    if include_zero_freq:
        variance = total_sq / nindep
    else:
        ndofs = ndofs[1:]
        freqs = freqs[1:]
        amplitudes = amplitudes[1:]
        variance = total_sq / nindep - mean**2
        variance *= (nstep * nindep) / (nstep * nindep - 1)

    return Spectrum(mean, variance, timestep, nstep, freqs, ndofs, amplitudes)


def _process_sequences(
    sequences: ArrayLike, prefactors: ArrayLike, nstep: int | None
) -> tuple[int, NDArray[float], float, float]:
    """Process a batch of sequences and compute the spectrum.

    Parameters
    ----------
    sequences
        The input sequences, which can be in two forms:
        This is an array with shape ``(nindep, nstep)`` or ``(nstep,)``.
        In case of a 2D array, each row is a time-dependent sequence.
        In case of a 1D array, a single sequence is used.
    prefactors
        A factor to be multiplied with the autocorrelation function
        to give it a physically meaningful unit.
        If multiple prefactors are given, each one is multiplied with the corresponding sequence.
    nstep
        The expected number of time steps in the input sequences or None.
        This is only used for consistency checking.

    Returns
    -------
    nstep
        The number of time steps in the input.
    nindep
        The number of independent sequences.
    amplitudes
        The spectrum amplitudes averaged over the given input sequences.
    total
        The sum of the input sequences.
    total_sq
        The sum of the squares of the input sequences.
    """
    sequences = np.asarray(sequences)
    prefactors = np.asarray(prefactors)

    # Handle single sequence case
    if sequences.ndim == 1:
        sequences = sequences.reshape(1, -1)
    elif sequences.ndim != 2:
        raise ValueError("Sequences must be a 1D or 2D array.")

    if prefactors.ndim == 0:
        prefactors = np.full(sequences.shape[0], prefactors)

    # Get basic parameters of the input sequences.
    if nstep is None:
        nstep = sequences.shape[1]
    elif nstep != sequences.shape[1]:
        raise ValueError("All sequences must have the same length.")
    nindep = sequences.shape[0]

    # Multiply the square root of (prefactor / 2) with the sequences.
    sequences = np.sqrt(prefactors)[:, None] * sequences

    # Compute the spectrum.
    # We already divide by nstep here to keep the order of magnitude under control.
    amplitudes = 0.5 * (abs(np.fft.rfft(sequences, axis=1)) ** 2).sum(axis=0) / nstep

    # Compute the variance of the input sequences.
    total = sequences.sum() / nstep
    total_sq = np.linalg.norm(sequences) ** 2 / nstep

    return nstep, nindep, amplitudes, total, total_sq
