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

import numpy as np
import pytest
from conftest import check_gradient, check_hessian

from stacie.model import ExpTailModel, WhiteNoiseModel

FREQS = np.linspace(0, 0.5, 10)
TIMESTEP = 1.2

PARS_REF_EXP_TAIL = [
    [1.2, 0.9, 1.1],
    [-0.3, 0.5, 1.4],
    [0.1, -7.7, 1.4],
    [-3.1, 0.0, 1.6],
    [0.0, 0.0, 1.5],
    [0.0, 15.0, 0.6],
    [-108.0, -77.7, 1.5],
    [108.0, 70.7, 1.2],
]


@pytest.mark.parametrize("pars_ref", PARS_REF_EXP_TAIL)
def test_gradient_exptail(pars_ref):
    pars_ref = np.array(pars_ref)
    model = ExpTailModel()
    check_gradient(lambda pars, deriv=0: model.compute(FREQS, TIMESTEP, pars, deriv), pars_ref)


@pytest.mark.parametrize("pars_ref", PARS_REF_EXP_TAIL)
def test_hessian_exptail(pars_ref):
    pars_ref = np.array(pars_ref)
    model = ExpTailModel()
    check_hessian(lambda pars, deriv=0: model.compute(FREQS, TIMESTEP, pars, deriv), pars_ref)


PARS_REF_WHITE = [[-102.0], [20.0], [-0.1], [0.0], [5.5]]


@pytest.mark.parametrize("pars_ref", PARS_REF_WHITE)
def test_gradient_white(pars_ref):
    pars_ref = np.array(pars_ref)
    model = WhiteNoiseModel()
    check_gradient(lambda pars, deriv=0: model.compute(FREQS, TIMESTEP, pars, deriv), pars_ref)


@pytest.mark.parametrize("pars_ref", PARS_REF_WHITE)
def test_hessian_white(pars_ref):
    pars_ref = np.array(pars_ref)
    model = WhiteNoiseModel()
    check_hessian(lambda pars, deriv=0: model.compute(FREQS, TIMESTEP, pars, deriv), pars_ref)
