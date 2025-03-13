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
"""The Stacie package."""

from __future__ import annotations

from .estimate import estimate_acint
from .plot import UnitConfig, plot_results
from .spectrum import compute_spectrum
from .summary import summarize_results

__all__ = (
    "UnitConfig",
    "__version__",
    "__version_tuple__",
    "compute_spectrum",
    "estimate_acint",
    "plot_results",
    "summarize_results",
)


try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0a-dev"
    __version_tuple__ = (0, 0, 0, "a-dev")
