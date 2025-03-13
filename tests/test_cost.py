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
"""Tests for ``stacie.cost``."""

import numdifftools as nd
import numpy as np
import pytest
from conftest import check_curv, check_deriv, check_gradient, check_hessian

from stacie.cost import LowFreqCost, logpdf_gamma
from stacie.model import ExpTailModel

LOGPDF_GAMMA_CASES = [
    (1.0, 1.0, 2.0),
    [1.2, 0.9, 2.2],
    [0.3, 0.001, 0.7],
    ([0.4, 1.1, 1.5], [1.0, 2.0, 3.0], [1.2, 4.3, 8.1]),
]


@pytest.mark.parametrize(("x", "kappa", "theta_ref"), LOGPDF_GAMMA_CASES)
def test_logpdf_gamma_deriv1(x, kappa, theta_ref):
    check_deriv(lambda theta, deriv=0: logpdf_gamma(x, kappa, theta, deriv), theta_ref)


@pytest.mark.parametrize(("x", "kappa", "theta_ref"), LOGPDF_GAMMA_CASES)
def test_logpdf_gamma_deriv2(x, kappa, theta_ref):
    check_curv(lambda theta, deriv=0: logpdf_gamma(x, kappa, theta, deriv), theta_ref)


@pytest.fixture
def mycost():
    timestep = 1.3
    freqs = np.linspace(0, 0.5 / timestep, 10)
    model = ExpTailModel()
    amplitudes = np.array([1.5, 1.4, 1.1, 0.9, 0.8, 1.0, 0.9, 0.9, 0.8, 1.1])
    ndofs = np.array([5, 10, 10, 10, 10, 10, 10, 10, 10, 5])
    return LowFreqCost(timestep, freqs, amplitudes, ndofs, model)


PARS_REF_EXP_TAIL = [
    [1.2, 0.9, 2.2],
    [3.0, 0.1, 2.5],
    [0.1, 4.0, 2.7],
    [108.0, 77.7, 1.8],
]


@pytest.mark.parametrize("pars_ref", PARS_REF_EXP_TAIL)
def test_gradient_exptail(mycost, pars_ref):
    check_gradient(mycost, pars_ref)


@pytest.mark.parametrize("pars_ref", PARS_REF_EXP_TAIL)
def test_hessian_exptail(mycost, pars_ref):
    check_hessian(mycost, pars_ref)


@pytest.mark.parametrize("pars_ref", PARS_REF_EXP_TAIL)
def test_cost_grad_sensitivity(mycost, pars_ref):
    amplitudes0 = mycost.amplitudes.copy()
    sensitivity = mycost.props(pars_ref, deriv=2)["cost_grad_sensitivity"]

    def cost_grad(amplitudes):
        mycost.amplitudes = amplitudes
        return mycost(pars_ref, deriv=1)[1]

    num_sensitivity, info = nd.Gradient(cost_grad, full_output=True)(amplitudes0)
    error = np.clip(info.error_estimate, 1e-15, np.inf)
    assert sensitivity / error == pytest.approx(num_sensitivity / error, abs=100)
