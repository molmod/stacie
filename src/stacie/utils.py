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
"""Utilities for preparing inputs."""

import attrs
import numpy as np
from numpy.typing import NDArray

__all__ = (
    "PositiveDefiniteError",
    "UnitConfig",
    "block_average",
    "label_unit",
    "mixture_stats",
    "robust_dot",
    "robust_posinv",
    "split",
)


@attrs.define
class UnitConfig:
    """Unit configuration for plotting function.

    Note that values are *divided* by their units before plotting.
    This class only affects screen output and plotting.
    It never influences numerical values in STACIE's computations.
    """

    acint_symbol: str = attrs.field(default=r"\mathcal{I}", kw_only=True)
    """The symbol used for the autocorrelation integral."""

    acint_unit_str: str = attrs.field(default="", kw_only=True)
    """The text used for the autocorrelation integral unit."""

    acint_unit: float = attrs.field(default=1.0, kw_only=True)
    """The unit of an autocorrelation integral."""

    acint_fmt: str = attrs.field(default=".2e", kw_only=True)
    """The format string for an autocorrelation integral."""

    freq_unit_str: str = attrs.field(default="", kw_only=True)
    """The text used for the frequency unit."""

    freq_unit: float = attrs.field(default=1.0, kw_only=True)
    """The unit of frequency."""

    freq_fmt: str = attrs.field(default=".2e", kw_only=True)
    """The format string for a frequency."""

    time_unit_str: str = attrs.field(default="", kw_only=True)
    """The text used for the time unit."""

    time_unit: float = attrs.field(default=1.0, kw_only=True)
    """The unit of time."""

    time_fmt: str = attrs.field(default=".2e", kw_only=True)
    """The format string for a time value."""

    clevel: float = attrs.field(default=0.95)
    """The confidence level used to plot confidence intervals."""

    @property
    def clb(self) -> float:
        """The confidence lower bound used to plot confidence intervals."""
        return (1 - self.clevel) / 2

    @property
    def cub(self) -> float:
        """The confidence upper bound used to plot confidence intervals."""
        return (1 + self.clevel) / 2


def label_unit(label: str, unit_str: str | None) -> str:
    """Format a label with the unit string as `label [unit]`.

    When the unit is ``""`` or ``None``, the unit is omitted.

    Parameters
    ----------
    label
        The label text.
    unit_str
        The unit string.
    """
    if unit_str in ("", None):
        return label
    return f"{label} [{unit_str}]"


def split(sequences: NDArray[float], nsplit: int) -> NDArray:
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


def block_average(sequences: NDArray[float], size: int) -> NDArray:
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


class PositiveDefiniteError(ValueError):
    """Raised when a matrix is not positive definite."""


def robust_posinv(matrix: NDArray[float]) -> tuple[NDArray, NDArray, NDArray, NDArray]:
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
        raise PositiveDefiniteError("matrix is not square.")
    if not np.isfinite(matrix).all():
        raise PositiveDefiniteError("matrix contains NaN or inf.")
    matrix = 0.5 * (matrix + matrix.T)
    diag = np.diag(matrix)
    if diag.min() <= 0:
        raise PositiveDefiniteError(f"matrix has nonpositive diagonal elements: {diag=}.")
    scales = np.sqrt(np.diag(matrix))
    scaled_matrix = (matrix / scales[:, None]) / scales
    evals, evecs = np.linalg.eigh(scaled_matrix)
    if evals.min() <= 0:
        raise PositiveDefiniteError(f"matrix has non-positive eigenvalues: {evals=}")
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


def mixture_stats(means: NDArray[float], covars: NDArray[float], weights: NDArray[float]):
    """Compute the statistics of the (Gaussian) mixture distribution.

    Parameters
    ----------
    means
        The means of the mixture components.
        Weighted averages are taken over the first index.
        Shape is ``(ncomp, nfeature)`` or ``(ncomp,)``.
        If the shape is ``(ncomp,)``, the means are interpreted as scalars.
        If the shape is ``(ncomp, nfeature)``, the means are interpreted as vectors.
    covars
        The covariances of the mixture components.
        If the shape matches that of the ``means`` argument,
        this array is interpreted as a diagonal covariance matrix.
        If the shape is ``(ncomp, nfeature, nfeature)``,
        this array is interpreted as full covariance matrices.
    weights
        The weights of the mixture components.
        Shape is ``(ncomp,)``.
        The weights are normalized to sum to 1.

    Returns
    -------
    mean
        The mean of the mixture distribution.
        Shape is ``(nfeature,)``.
    covar
        If the input covariance matrix is diagonal, the output covariance matrix
        is also diagonal and has shape ``(nfeature,)``.
        If the input covariance matrix is full, the output covariance matrix
        is also full and has shape ``(nfeature, nfeature)``.
    """
    means = np.asarray(means)
    covars = np.asarray(covars)
    weights = np.asarray(weights)
    if means.ndim < 1:
        raise ValueError("means must be at least a 1D array.")
    unpack = False
    if means.ndim == 1:
        means = means.reshape(-1, 1)
        unpack = True
    ncomp, nfeature = means.shape
    if covars.ndim < 1:
        raise ValueError("covars must be at least a 1D array.")
    if covars.ndim == 1:
        covars = covars.reshape(-1, 1)
    if weights.shape != (ncomp,):
        raise ValueError("weights must be a 1D vector with ncomp elements.")
    weights = weights / weights.sum()
    mean = np.dot(weights, means)
    deltas = means - mean
    if covars.shape == means.shape:
        covar = np.dot(weights, covars)
        covar += np.einsum("i,ij,ij->j", weights, deltas, deltas)
    elif covars.shape == (ncomp, nfeature, nfeature):
        covar = np.einsum("i,ijk->jk", weights, covars)
        covar += np.einsum("i,ij,ik->jk", weights, deltas, deltas)
    else:
        raise ValueError(
            f"Unsupported shape for covars: {covars.shape}. "
            f"Expected ({ncomp},), {means.shape}, or ({ncomp}, {nfeature}, {nfeature})."
        )
    if unpack:
        mean = mean[0]
        covar = covar[0]
    return mean, covar
