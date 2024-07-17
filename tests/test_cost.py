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

LOGPDF_GAMMA_CASES = [(1.0, 1.0, 2.0), ([0.4, 1.1, 1.5], [1.0, 2.0, 3.0], [1.2, 4.3, 8.1])]


@pytest.mark.parametrize(("x", "kappa", "theta0"), LOGPDF_GAMMA_CASES)
def test_logpdf_gamma_deriv1(x, kappa, theta0):
    def value(theta):
        return logpdf_gamma(x, kappa, theta)[0]

    def grad(theta):
        return logpdf_gamma(x, kappa, theta, 1)[1]

    assert_allclose(grad(theta0), nd.Derivative(value)(theta0), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(("x", "kappa", "theta0"), LOGPDF_GAMMA_CASES)
def test_logpdf_gamma_deriv2(x, kappa, theta0):
    def value(theta):
        return logpdf_gamma(x, kappa, theta, 1)[1]

    def grad(theta):
        return logpdf_gamma(x, kappa, theta, 2)[2]

    assert_allclose(grad(theta0), nd.Derivative(value)(theta0), atol=1e-12, rtol=1e-12)


@pytest.fixture()
def mycost():
    timestep = 1.3
    freqs = np.linspace(0, 0.5 / timestep, 10)
    model = ExpTailModel()
    amplitudes = np.array([1.5, 1.4, 1.1, 0.9, 0.8, 1.0, 0.9, 0.9, 0.8, 1.1])
    ndofs = np.array([5, 10, 10, 10, 10, 10, 10, 10, 10, 5])
    return LowFreqCost(timestep, freqs, amplitudes, ndofs, model)


def test_gradient_exptail(mycost):
    pars0 = np.array([1.2, 0.9, 2.2])

    def value(pars):
        return mycost.funcgrad(pars)[0]

    def grad(pars):
        return mycost.funcgrad(pars)[1]

    assert_allclose(grad(pars0), nd.Gradient(value)(pars0), atol=1e-12, rtol=1e-12)


def test_hessian_exptail(mycost):
    pars0 = np.array([1.2, 0.9, 2.2])

    def value(pars):
        return mycost.funcgrad(pars)[1]

    def grad(pars):
        return mycost.hess(pars)

    assert_allclose(grad(pars0), nd.Gradient(value)(pars0), atol=1e-12, rtol=1e-12)
