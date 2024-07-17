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
"""Tests for ``stacie.model``."""

import numdifftools as nd
import numpy as np
from numpy.testing import assert_allclose
from stacie.model import ExpTailModel, WhiteNoiseModel


def test_gradient_exptail():
    omegas = np.linspace(0, np.pi, 10)
    pars0 = np.array([1.2, 0.9, 2.2])
    model = ExpTailModel()

    def value(pars):
        return model(omegas, pars)[0]

    def grad(pars):
        return model(omegas, pars, 1)[1].T

    assert_allclose(grad(pars0), nd.Gradient(value)(pars0), atol=1e-12, rtol=1e-12)


def test_hessian_exptail():
    omegas = np.linspace(0, np.pi, 10)
    pars0 = np.array([1.2, 0.9, 2.2])
    model = ExpTailModel()

    def value(pars):
        return model(omegas, pars, 1)[1].T

    def grad(pars):
        return model(omegas, pars, 2)[2].transpose(2, 0, 1)

    assert_allclose(grad(pars0), nd.Gradient(value)(pars0), atol=1e-12, rtol=1e-12)


def test_gradient_white():
    omegas = np.linspace(0, np.pi, 10)
    pars0 = np.array([1.2])
    model = WhiteNoiseModel()

    def value(pars):
        return model(omegas, pars)[0]

    def grad(pars):
        return model(omegas, pars, 1)[1][0]

    assert_allclose(grad(pars0), nd.Gradient(value)(pars0), atol=1e-12, rtol=1e-12)


def test_hessian_white():
    omegas = np.linspace(0, np.pi, 10)
    pars0 = np.array([1.2])
    model = WhiteNoiseModel()

    def value(pars):
        return model(omegas, pars, 1)[1].T

    def grad(pars):
        return model(omegas, pars, 2)[2][0, 0]

    assert_allclose(grad(pars0), nd.Gradient(value)(pars0), atol=1e-12, rtol=1e-12)
