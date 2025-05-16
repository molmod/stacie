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
"""Test configuration."""

import matplotlib as mpl
import numdifftools as nd
import numpy as np
import pytest

__all__ = ("check_curv", "check_deriv", "check_gradient", "check_hessian")

mpl.rcParams["backend"] = "Agg"
mpl.rcParams["font.size"] = 8.0
mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False


def check_deriv(func, x0):
    """Check the derivative with STACIE's convention for returning gradients."""
    deriv = func(x0, deriv=1)[1]
    num_deriv, info = nd.Derivative(lambda x: func(x)[0], full_output=True)(x0)
    error = np.clip(info.error_estimate, 1e-15, np.inf)
    assert deriv / error == pytest.approx(num_deriv / error, abs=100)


def check_curv(func, x0):
    """Check the curvature with STACIE's convention for returning gradients."""
    curv = func(x0, deriv=2)[2]
    num_curv, info = nd.Derivative(lambda x: func(x, deriv=1)[1], full_output=True)(x0)
    error = np.clip(info.error_estimate, 1e-15, np.inf)
    assert curv / error == pytest.approx(num_curv / error, abs=100)


def check_gradient(func, x0):
    """Check the gradient with STACIE's convention for returning gradients."""
    grad = func(x0, deriv=1)[1]
    num_grad, info = nd.Gradient(lambda x: func(x)[0], full_output=True)(x0)
    error = np.clip(info.error_estimate, 1e-15, np.inf)
    if num_grad.ndim == 2:
        num_grad = num_grad.T
        error = error.T
    assert grad / error == pytest.approx(num_grad / error, abs=100)


def check_hessian(func, x0):
    """Check the Hessian with STACIE's convention for returning gradients."""
    hess = func(x0, deriv=2)[2]
    num_hess, info = nd.Gradient(lambda x: func(x, deriv=1)[1], full_output=True)(x0)
    error = np.clip(info.error_estimate, 1e-15, np.inf)
    assert hess / error == pytest.approx(num_hess / error, abs=100)
