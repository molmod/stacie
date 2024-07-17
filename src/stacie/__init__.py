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
"""The Stacie package."""

from .estimate import estimate_acfint
from .plot import UnitConfig, plot_results
from .spectrum import prepare_acfint

__all__ = (
    "__version__",
    "__version_tuple__",
    "prepare_acfint",
    "estimate_acfint",
    "plot_results",
    "UnitConfig",
)


try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0a-dev"
    __version_tuple__ = (0, 0, 0, "a-dev")
