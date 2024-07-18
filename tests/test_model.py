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
import pytest
from numpy.testing import assert_allclose
from stacie.model import ExpTailModel, WhiteNoiseModel

OMEGAS = np.linspace(0, np.pi, 10)

PARS_REF_EXP_TAIL = [
    [1.2, 0.9, 2.2],
    [-0.3, 0.5, 0.7],
    [0.1, -7.7, 0.7],
    [-3.1, 0.0, 1.6],
    [0.0, 0.0, 2.6],
    [0.0, 15.0, 1.2],
    [-108.0, -77.7, 5.0],
    [108.0, 77.7, 3.6],
]


@pytest.mark.parametrize("pars_ref", PARS_REF_EXP_TAIL)
def test_gradient_exptail(pars_ref):
    pars_ref = np.array(pars_ref)
    model = ExpTailModel()
    assert_allclose(
        model(OMEGAS, pars_ref, 1)[1].T,
        nd.Gradient(lambda pars: model(OMEGAS, pars)[0])(pars_ref),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("pars_ref", PARS_REF_EXP_TAIL)
def test_hessian_exptail(pars_ref):
    pars_ref = np.array(pars_ref)
    model = ExpTailModel()
    assert_allclose(
        model(OMEGAS, pars_ref, 2)[2].transpose(2, 0, 1),
        nd.Gradient(lambda pars: model(OMEGAS, pars, 1)[1].T)(pars_ref),
        atol=1e-12,
        rtol=1e-12,
    )


PARS_REF_WHITE = [[-102.0], [20.0], [-0.1], [0.0], [5.5]]


@pytest.mark.parametrize("pars_ref", PARS_REF_WHITE)
def test_gradient_white(pars_ref):
    pars_ref = np.array(pars_ref)
    model = WhiteNoiseModel()
    assert_allclose(
        model(OMEGAS, pars_ref, 1)[1][0],
        nd.Gradient(lambda pars: model(OMEGAS, pars)[0])(pars_ref),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("pars_ref", PARS_REF_WHITE)
def test_hessian_white(pars_ref):
    pars_ref = np.array(pars_ref)
    model = WhiteNoiseModel()
    assert_allclose(
        model(OMEGAS, pars_ref, 2)[2][0, 0],
        nd.Gradient(lambda pars: model(OMEGAS, pars, 1)[1].T)(pars_ref),
        atol=1e-12,
        rtol=1e-12,
    )
