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
"""Utility to prepare spectrum and other inputs for given time series."""

import attrs
import numpy as np
from numpy.typing import NDArray

__all__ = ("Spectrum", "prepare_acfint")


@attrs.define
class Spectrum:
    """Container class holding all the inputs for the ACF integral esstimate."""

    ndofs: NDArray[int] = attrs.field()
    """The number of independnt contributions to each amplitude."""

    prefactor: float = attrs.field()
    """The given prefactor for the spectrum to get the right units for the ACF integral."""

    times: NDArray[float] = attrs.field()
    """The equidistant time axis of the sequences, always starts at zero."""

    freqs: NDArray[float] = attrs.field()
    """The equidistant frequency axis of the spectrum."""

    amplitudes: NDArray[float] = attrs.field()
    """The spectrum amplitudes averaged over the given input sequences."""

    amplitudes_ref: NDArray[float] | None = attrs.field(default=None)
    """Optionally, the known analytical model of the power spectrum, on the same frequency grid."""

    @property
    def nstep(self) -> int:
        """The number of time steps of the input sequences."""
        return len(self.times)

    @property
    def timestep(self) -> float:
        """The time span between two subsequent items in the given time series."""
        return self.times[1] - self.times[0]

    @property
    def nfreq(self) -> int:
        """The number of irfft frequency grid points."""
        return len(self.freq)


def prepare_acfint(
    sequences: NDArray[float], *, prefactor: float = 0.5, timestep: float = 1
) -> Spectrum:
    """Compute a spectrum and store all physical inputs in a ``Spectrum`` instance.

    Parameters
    ----------
    sequences
        The input sequences, array with shape `(nindep, nstep)`,
        of which each row is a time-dependent sequence.
        All sequences are assumed to be statistically independent.
        The time step is assumed to be 1.
        If needed, multiply the resulting acfint and acfint_err with the appropriate time step.
    prefactor
        A factor to be multiplied with the autocorrelation function
        to give it a physically meaningful unit.
    timestep
        The timestep of the input sequence.

    Returns
    -------
    spectrum
        An instance of the ``Spectrum`` object holding all the inputs needed to estimate
        the integral of the autocorrelation function.
    """
    # Get basic parameters of the input sequences.
    nindep, nstep = sequences.shape
    times = np.arange(nstep) * timestep
    freqs = np.fft.rfftfreq(nstep, d=timestep)

    # Number of "degrees of freedom" (contributions) to each amplitude
    ndofs = np.full(freqs.shape, 2 * nindep)
    ndofs[0] = nindep
    if len(freqs) % 2 == 0:
        ndofs[-1] = nindep

    # Compute the spectrum and scale it.
    amplitudes = (abs(np.fft.rfft(sequences, axis=1)) ** 2).mean(axis=0)
    amplitudes *= prefactor * timestep / nstep

    return Spectrum(ndofs, prefactor, times, freqs, amplitudes)
