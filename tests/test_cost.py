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
"""Tests for ``stacie.cost``."""

import numdifftools as nd
import numpy as np
import pytest
from numpy.testing import assert_allclose
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
    assert_allclose(
        logpdf_gamma(x, kappa, theta_ref, 1)[1],
        nd.Derivative(lambda theta: logpdf_gamma(x, kappa, theta)[0])(theta_ref),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize(("x", "kappa", "theta_ref"), LOGPDF_GAMMA_CASES)
def test_logpdf_gamma_deriv2(x, kappa, theta_ref):
    assert_allclose(
        logpdf_gamma(x, kappa, theta_ref, 2)[2],
        nd.Derivative(lambda theta: logpdf_gamma(x, kappa, theta, 1)[1])(theta_ref),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.fixture()
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
    [108.0, 77.7, 3.6],
]


@pytest.mark.parametrize("pars_ref", PARS_REF_EXP_TAIL)
def test_gradient_exptail(mycost, pars_ref):
    pars_ref = np.array(pars_ref)
    assert_allclose(
        mycost.funcgrad(pars_ref)[1],
        nd.Gradient(lambda pars: mycost.funcgrad(pars)[0])(pars_ref),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("pars_ref", PARS_REF_EXP_TAIL)
def test_hessian_exptail(mycost, pars_ref):
    pars_ref = np.array(pars_ref)
    assert_allclose(
        mycost.hess(pars_ref),
        nd.Gradient(lambda pars: mycost.funcgrad(pars)[1])(pars_ref),
        atol=1e-12,
        rtol=1e-12,
    )
