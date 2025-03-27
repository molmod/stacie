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

__all__ = ("PostiveDefiniteError", "block_average", "robust_dot", "robust_posinv", "split")


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


class PostiveDefiniteError(ValueError):
    """Raised when a matrix is not positive definite."""


def robust_posinv(matrix: ArrayLike) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Compute the eigenvalues, eigenvectors and inverse of a positive definite symmetric matrix.

    This function is a robust version of ``numpy.linalg.eigh`` and ``numpy.linalg.inv``
    that can handle large variations in order of magnitude of the diagonal elements.
    If the matrix is not positive definite, a ``ValueError`` is raised.

    Parameters
    ----------
    matrix
        Input matrix to be diagonalized.

    Returns
    -------
    scales
        The scales used to precondition the matrix.
    evals
        The eigenvalues of the preconditioned matrix.
    evecs
        The eigenvectors of the preconditioned matrix.
    inverse
        The inverse of the original.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise PostiveDefiniteError("Input matrix must be a square matrix.")
    if not np.isfinite(matrix).all():
        raise PostiveDefiniteError("Matrix must not contain NaN or inf.")
    matrix = 0.5 * (matrix + matrix.T)
    if np.diag(matrix).min() <= 0:
        raise PostiveDefiniteError(
            "Matrix must be positive definite but has nonpositive diagonal elements."
        )
    scales = np.sqrt(np.diag(matrix))
    scaled_matrix = (matrix / scales[:, None]) / scales
    evals, evecs = np.linalg.eigh(scaled_matrix)
    if evals.min() <= 0:
        raise PostiveDefiniteError("Matrix is not positive definite.")
    # Construct matrix square root of inverse first, to guarantee that the result is symmetric.
    half = evecs / np.sqrt(evals)
    scaled_inverse = np.dot(half, half.T)
    inverse = (scaled_inverse / scales[:, None]) / scales
    return scales, evals, evecs, inverse


def robust_dot(scales, evals, evecs, other):
    """Compute the dot product of a robustly diagonalized matrix with another matrix.

    - The first three arguments are the output of ``robust_posinv``.
    - To multiply with the inverse, just use element-wise inversion of ``scales`` and ``evals``.

    Parameters
    ----------
    scales
        The scales used to precondition the matrix.
    evals
        The eigenvalues of the preconditioned matrix.
    evecs
        The eigenvectors of the preconditioned matrix.
    other
        The other matrix to be multiplied. 1D or 2D arrays are accepted.

    Returns
    -------
    result
        The result of the dot product.
    """
    if other.ndim == 2:
        scales = scales[:, None]
        evals = evals[:, None]
    return np.dot(evecs, np.dot(evecs.T, other * scales) * evals) * scales
