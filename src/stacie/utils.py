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
"""Utilities for preparing inputs."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ("split_sequences",)


def split_sequences(sequences: ArrayLike, nsplit: int) -> NDArray:
    sequences = np.asarray(sequences)
    if sequences.ndim == 1:
        sequences.shape = (1, -1)
    length = sequences.shape[1] // nsplit
    return sequences[:, : length * nsplit].reshape(-1, length)
