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
"""Tests for ``stacie.model``"""

import numpy as np
import pytest
from conftest import check_gradient, check_hessian

from stacie.model import ChebyshevModel, ExpTailModel, PadeModel, guess

FREQS = np.linspace(0, 0.5, 10)
AMPLITUDES_REF = np.linspace(2, 1, 10)
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
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    check_gradient(lambda pars, deriv=0: model.compute(TIMESTEP, FREQS, pars, deriv), pars_ref)


@pytest.mark.parametrize("pars_ref", PARS_REF_EXP_TAIL)
def test_hessian_exptail(pars_ref):
    pars_ref = np.array(pars_ref)
    model = ExpTailModel()
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    check_hessian(lambda pars, deriv=0: model.compute(TIMESTEP, FREQS, pars, deriv), pars_ref)


def test_guess_exptail():
    model = ExpTailModel()
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    rng = np.random.default_rng(734)
    amplitudes = rng.normal(size=len(FREQS)) ** 2
    ndofs = np.full(len(FREQS), 2)
    pars_init = guess(model, TIMESTEP, FREQS, ndofs, amplitudes, model.par_scales, rng, 10)
    assert len(pars_init) == 3
    assert np.isfinite(pars_init).all()


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_cheb_npar(degree):
    model = ChebyshevModel(degree)
    assert model.npar == degree + 1


PARS_REF_CHEBY = [
    [-12.0, 3.4, 78.3],
    [9.0, 8.1],
    [-0.1, 0.02, -0.7, 0.3],
    [0.0, 0.0, 0.0],
    [0.0, 3.0],
]


@pytest.mark.parametrize("pars_ref", PARS_REF_CHEBY)
def test_gradient_cheb(pars_ref):
    pars_ref = np.array(pars_ref)
    model = ChebyshevModel(len(pars_ref) - 1)
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    check_gradient(lambda pars, deriv=0: model.compute(TIMESTEP, FREQS, pars, deriv), pars_ref)


@pytest.mark.parametrize("pars_ref", PARS_REF_CHEBY)
def test_hessian_cheb(pars_ref):
    pars_ref = np.array(pars_ref)
    model = ChebyshevModel(len(pars_ref) - 1)
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    check_hessian(lambda pars, deriv=0: model.compute(TIMESTEP, FREQS, pars, deriv), pars_ref)


def test_guess_cheb():
    model = ChebyshevModel(2)
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    rng = np.random.default_rng(123)
    amplitudes = rng.normal(size=len(FREQS)) ** 2
    ndofs = np.full(len(FREQS), 2)
    pars_init = guess(model, TIMESTEP, FREQS, ndofs, amplitudes, model.par_scales, rng, 10)
    assert len(pars_init) == 3
    assert np.isfinite(pars_init).all()


@pytest.mark.parametrize(
    ("degree", "npar"),
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 3),
        (5, 3),
        (6, 4),
        (7, 4),
    ],
)
def test_even_cheb_npar(degree, npar):
    model = ChebyshevModel(degree, even=True)
    assert model.npar == npar


PARS_REF_EVEN_CHEBY = [
    [1.5],
    [0.0],
    [0.0, 0.0],
    [2.31, 3.71],
    [0.0, 0.0, 0.0],
    [-2.0, 0.4, -8.3],
]


@pytest.mark.parametrize("pars_ref", PARS_REF_EVEN_CHEBY)
def test_gradient_cheb_even(pars_ref):
    pars_ref = np.array(pars_ref)
    model = ChebyshevModel(2 * len(pars_ref) - 2, even=True)
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    check_gradient(lambda pars, deriv=0: model.compute(TIMESTEP, FREQS, pars, deriv), pars_ref)


@pytest.mark.parametrize("pars_ref", PARS_REF_EVEN_CHEBY)
def test_hessian_cheb_even(pars_ref):
    pars_ref = np.array(pars_ref)
    model = ChebyshevModel(2 * len(pars_ref) - 2, even=True)
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    check_hessian(lambda pars, deriv=0: model.compute(TIMESTEP, FREQS, pars, deriv), pars_ref)


def test_guess_cheb_even():
    model = ChebyshevModel(4, even=True)
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    rng = np.random.default_rng(123)
    amplitudes = rng.normal(size=len(FREQS)) ** 2
    ndofs = np.full(len(FREQS), 2)
    pars_init = guess(model, TIMESTEP, FREQS, ndofs, amplitudes, model.par_scales, rng, 10)
    assert len(pars_init) == 3
    assert np.isfinite(pars_init).all()


def test_pade_npar():
    assert PadeModel([0, 2], [2]).npar == 3


PARS_REF_PADE = [
    [0.0, 0.0, 0.0, 0.0],
    [0.5, -2.0, 0.4, -8.3],
    [1.3, 2.0, -0.4, 0.0],
    [0.0, 0.5, 3.2, 1.3],
]


@pytest.mark.parametrize("pars_ref", PARS_REF_PADE)
def test_gradient_pade(pars_ref):
    pars_ref = np.array(pars_ref)
    model = PadeModel([0, 1, 2], [2])
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    check_gradient(lambda pars, deriv=0: model.compute(TIMESTEP, FREQS, pars, deriv), pars_ref)


@pytest.mark.parametrize("pars_ref", PARS_REF_PADE)
def test_hessian_pade(pars_ref):
    pars_ref = np.array(pars_ref)
    model = PadeModel([0, 1, 2], [2])
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    check_hessian(lambda pars, deriv=0: model.compute(TIMESTEP, FREQS, pars, deriv), pars_ref)


def test_guess_pade():
    model = PadeModel([0, 2], [2])
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    rng = np.random.default_rng(123)
    amplitudes = rng.normal(size=len(FREQS)) ** 2
    ndofs = np.full(len(FREQS), 2)
    pars_init = guess(model, TIMESTEP, FREQS, ndofs, amplitudes, model.par_scales, rng, 10)
    assert len(pars_init) == 3
    assert np.isfinite(pars_init).all()


def test_guess_pade_detailed():
    freqs = np.linspace(0, 1.0, 10)
    model = PadeModel([0, 2], [2])
    model.configure_scales(TIMESTEP, freqs, AMPLITUDES_REF)
    pars_ref = np.array([3.0, 1.5, 2.0])
    amplitudes_ref = model.compute(1.0, freqs, pars_ref, 0)[0]
    x = freqs / freqs[-1]
    assert amplitudes_ref == pytest.approx((3.0 + 1.5 * x**2) / (1.0 + 2.0 * x**2), rel=1e-10)
    ndofs = np.full(len(freqs), 20)
    pars_init_low, amplitudes_low = model.solve_linear(1.0, freqs, ndofs, amplitudes_ref, [])
    assert pars_init_low == pytest.approx(pars_ref, rel=1e-10)
    assert amplitudes_low == pytest.approx(amplitudes_ref, rel=1e-10)
    par_scales = np.ones(3)
    rng = np.random.default_rng(123)
    pars_init = guess(model, 1.0, freqs, ndofs, amplitudes_ref, par_scales, rng, 10)
    assert pars_ref == pytest.approx(pars_init, rel=1e-10)
