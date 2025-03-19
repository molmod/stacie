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

from collections.abc import Iterable
from typing import Self

import attrs
import numpy as np
from numpy.typing import NDArray

from .utils import split

__all__ = ("Spectrum", "compute_spectrum")


@attrs.define
class Spectrum:
    """Container class holding all the inputs for the autocorrelation integral estimate."""

    prefactor: float = attrs.field(converter=float)
    """The given prefactor for the spectrum to fix the units for the autocorrelation integral."""

    mean: float = attrs.field(converter=float)
    """The mean of the input sequences."""

    variance: float = attrs.field(converter=float)
    """The variance of the input sequences."""

    timestep: float = attrs.field(converter=float)
    """The time between two subsequent elements in the given sequence."""

    nstep: int = attrs.field(converter=int)
    """The number of time steps in the input."""

    freqs: NDArray[float] = attrs.field()
    """The equidistant frequency axis of the spectrum."""

    ndofs: NDArray[int] = attrs.field()
    """The number of independnt contributions to each amplitude."""

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
    prefactor: float = 0.5,
    timestep: float = 1,
    include_zero_freq: bool = True,
    nsplit: int = 1,
) -> Spectrum:
    r"""Compute a spectrum and store all inputs for ``estimate_acint`` in a ``Spectrum`` instance.

    The spectrum amplitudes are computed as follows:

    .. math::

        C_k = \frac{F h}{N} \frac{1}{M}\sum_{m=1}^M \left|
            \sum_{n=0}^{N-1} x^{(m)}_n \exp\left(-i \frac{2 \pi n k}{N}\right)
        \right|^2

    where:

    - :math:`F` is the given prefactor,
    - :math:`h` is the timestep,
    - :math:`N` is the number of time steps in the input sequences,
    - :math:`M` is the number of independent sequences,
    - :math:`x^{(m)}_n` is the value of the :math:`m`-th sequence at time step :math:`n`,
    - :math:`k` is the frequency index.

    The sum over :math:`m` simply averages spectra obtained from different sequences.
    The factor :math:`F h/N` normalizes the spectrum so that its zero-frequency limit
    is an estimate of the autocorrelation integral.

    Parameters
    ----------
    sequences
        The input sequences, which can be in two forms:

        - An array with shape ``(nindep, nstep)`` or ``(nstep,)``.
          In case of a 2D array, each row is a time-dependent sequence.
          In case of a 1D array, a single sequence is used.
        - An iterable whose items are arrays as described in the previous point.
          This option is convenient when a single array does not fit in memory.

        All sequences are assumed to be statistically independent and have length ``nstep``.
        (Time correlations within one sequence are fine, obviously.)
        We recommend using multiple independent sequences to reduce uncertainties.
    prefactor
        A factor to be multiplied with the autocorrelation function
        to give it a physically meaningful unit.
    timestep
        The time step of the input sequence.
    include_zero_freq
        When set to False, the DC component of the spectrum is discarded.
    nsplit
        If larger than 1, the sequences are split into ``nsplit`` chunks
        and the spectrum of each chunk is computed separately, as if they are separate sequences.
        This reduces the resolution of the frequency axis and the variance of the spectrum.
        The function ``stacie.utils.split`` is used for this purpose.

    Returns
    -------
    spectrum
        A ``Spectrum`` object holding all the inputs needed to estimate
        the integral of the autocorrelation function.
    """
    # Handle single-array case
    if isinstance(sequences, np.ndarray):
        sequences = [sequences]

    # Process iterable of arrays
    if isinstance(sequences, Iterable):
        nindep = 0
        nstep = None
        amplitudes = 0
        total = 0
        total_sq = 0
        for data in sequences:
            array = np.asarray(data)

            # Handle single sequence case
            if array.ndim == 1:
                array = array.reshape(1, -1)
            elif array.ndim != 2:
                raise ValueError("Sequences must be a 1D or 2D array.")

            # Split the array into chunks for better statistics if requested.
            if nsplit > 1:
                array = split(array, nsplit)

            # Get basic parameters of the input sequences.
            if nstep is None:
                nstep = array.shape[1]
            elif nstep != array.shape[1]:
                raise ValueError("All sequences must have the same length.")
            nindep += array.shape[0]

            # Compute the spectrum.
            # We already divide by nstep here to keep the order of magnitude under control.
            amplitudes += (abs(np.fft.rfft(array, axis=1)) ** 2).sum(axis=0) / nstep

            # Compute the variance of the input sequences.
            total += array.sum() / nstep
            total_sq += np.linalg.norm(array) ** 2 / nstep
    else:
        raise TypeError("The sequence argument must be an array or an iterable of arrays.")

    # Frequency axis and scale of amplitudes
    freqs = np.fft.rfftfreq(nstep, d=timestep)
    amplitudes *= prefactor * timestep / nindep

    # Number of "degrees of freedom" (contributions) to each amplitude
    ndofs = np.full(freqs.shape, 2 * nindep)
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

    return Spectrum(prefactor, mean, variance, timestep, nstep, freqs, ndofs, amplitudes)
