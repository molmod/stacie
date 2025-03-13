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
"""Utilities for preparing inputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ("block_average", "split")


def split(sequences: ArrayLike, nsplit: int) -> NDArray:
    """Split input sequences into shorter parts of equal length.

    This reduces the resolution of the frequency axis of the spectrum,
    which may be useful when the sequence length is much longer than the exponential
    autocorrelation time.

    Parameters
    ----------
    sequences
        Input sequence(s) to be split, with shape ``(nseq, nstep)``.
        A single sequence with shape ``(nstep, )`` is also accepted.
    nsplit
        The number of splits.

    Returns
    -------
    split_sequences
        Splitted sequences, with shape ``(nseq * nsplit, nstep // nsplit)``.
    """
    sequences = np.asarray(sequences)
    if sequences.ndim == 1:
        sequences.shape = (1, -1)
    if not isinstance(nsplit, int) or nsplit <= 0 or nsplit > sequences.shape[-1] / 2:
        raise ValueError("nsplit must be a positive integer smaller than half the sequence length.")
    length = sequences.shape[1] // nsplit
    return sequences[:, : length * nsplit].reshape(-1, length)


def block_average(sequences: ArrayLike, size: int) -> NDArray:
    r"""Reduce input sequences by taking block averages.

    This reduces the maximum frequency of the frequency axis of the spectrum,
    which may be useful when the time step is much shorter than the exponential
    autocorrelation time.

    A time step :math:`h = \tau_\text{exp} / (20 \pi)` (after taking block averages)
    is recommended, not larger.

    Parameters
    ----------
    sequences
        Input sequence(s) to be block averaged, with shape ``(nseq, nstep)``.
        A single sequence with shape ``(nstep, )`` is also accepted.
    size
        The block size

    Returns
    -------
    blav_sequences
        Sequences of block averages, with shape ``(nseq, nstep // size)``
    """
    sequences = np.asarray(sequences)
    if sequences.ndim == 1:
        sequences.shape = (1, -1)
    length = sequences.shape[1] // size
    return sequences[:, : length * size].reshape(-1, length, size).mean(axis=2)
