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
"""Tests for ``stacie.model``"""

import numpy as np
import pytest
from conftest import check_gradient, check_hessian

from stacie.model import ExpPolyModel, PadeModel, guess

NFREQ = 10
FREQS = np.linspace(0, 0.5, NFREQ)
AMPLITUDES_REF = np.linspace(2, 1, NFREQ)
WEIGHTS = 1 - FREQS**2
TIMESTEP = 1.2


def check_vectorize_compute(model, pars_ref, broadcast=False):
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    pars_ref = np.array(pars_ref)
    amplitudes = model.compute(FREQS, pars_ref, deriv=2)
    nvec, npar = pars_ref.shape
    assert amplitudes[0].shape == (nvec, len(FREQS))
    if broadcast:
        assert amplitudes[1].shape == (1, npar, len(FREQS))
        assert amplitudes[2].shape == (1, npar, npar, len(FREQS))
    else:
        assert amplitudes[1].shape == (nvec, npar, len(FREQS))
        assert amplitudes[2].shape == (nvec, npar, npar, len(FREQS))
    for i, one_pars_ref in enumerate(pars_ref):
        one_amplitudes = model.compute(FREQS, one_pars_ref, deriv=2)
        assert one_amplitudes[0].shape == (len(FREQS),)
        assert one_amplitudes[1].shape == (npar, len(FREQS))
        assert one_amplitudes[2].shape == (npar, npar, len(FREQS))
        assert (one_amplitudes[0] == amplitudes[0][i]).all()
        if broadcast:
            assert (one_amplitudes[1] == amplitudes[1][0]).all()
            assert (one_amplitudes[2] == amplitudes[2][0]).all()
        else:
            assert (one_amplitudes[1] == amplitudes[1][i]).all()
            assert (one_amplitudes[2] == amplitudes[2][i]).all()


def check_vectorize_prior(model, pars_ref):
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    pars_ref = np.array(pars_ref)
    prior = model.neglog_prior(pars_ref, deriv=2)
    nvec, npar = pars_ref.shape
    assert prior[0].shape == (nvec,)
    assert prior[1].shape == (nvec, npar)
    assert prior[2].shape == (nvec, npar, npar)
    for i, one_pars_ref in enumerate(pars_ref):
        one_prior = model.neglog_prior(one_pars_ref, deriv=2)
        assert one_prior[0].shape == ()
        assert one_prior[1].shape == (npar,)
        assert one_prior[2].shape == (npar, npar)
        assert (one_prior[0] == prior[0][i]).all()
        assert (one_prior[1] == prior[1][i]).all()
        assert (one_prior[2] == prior[2][i]).all()


def test_exppoly_npar():
    assert ExpPolyModel([0, 2, 4]).npar == 3


PARS_REF_POLY = [
    [-12.0, 3.4, 78.3],
    [9.0, 8.1],
    [-0.1, 0.02, -0.7, 0.3],
    [0.0, 0.0, 0.0],
    [0.0, 3.0],
]


@pytest.mark.parametrize("npar", [2, 3])
def test_vectorize_exppoly_compute(npar: int):
    pars_ref = [p for p in PARS_REF_POLY if len(p) == npar]
    check_vectorize_compute(ExpPolyModel(list(range(npar))), pars_ref)


@pytest.mark.parametrize("npar", [2, 3])
def test_vectorize_exppoly_prior(npar: int):
    pars_ref = [p for p in PARS_REF_POLY if len(p) == npar]
    check_vectorize_prior(ExpPolyModel(list(range(npar))), pars_ref)


@pytest.mark.parametrize("pars_ref", PARS_REF_POLY)
def test_gradient_exppoly_compute(pars_ref):
    pars_ref = np.array(pars_ref)
    model = ExpPolyModel(list(range(len(pars_ref))))
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    check_gradient(lambda pars, *, deriv=0: model.compute(FREQS, pars, deriv=deriv), pars_ref)


@pytest.mark.parametrize("pars_ref", PARS_REF_POLY)
def test_hessian_exppoly_compute(pars_ref):
    pars_ref = np.array(pars_ref)
    model = ExpPolyModel(list(range(len(pars_ref))))
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    check_hessian(lambda pars, *, deriv=0: model.compute(FREQS, pars, deriv=deriv), pars_ref)


@pytest.mark.parametrize("pars_ref", PARS_REF_POLY)
def test_gradient_exppoly_prior(pars_ref):
    pars_ref = np.array(pars_ref)
    model = ExpPolyModel(list(range(len(pars_ref))))
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    check_gradient(lambda pars, *, deriv=0: model.neglog_prior(pars, deriv=deriv), pars_ref)


@pytest.mark.parametrize("pars_ref", PARS_REF_POLY)
def test_hessian_exppoly_prior(pars_ref):
    pars_ref = np.array(pars_ref)
    model = ExpPolyModel(list(range(len(pars_ref))))
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    check_hessian(lambda pars, *, deriv=0: model.neglog_prior(pars, deriv=deriv), pars_ref)


def test_guess_exppoly():
    model = ExpPolyModel([0, 1, 2])
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    rng = np.random.default_rng(123)
    amplitudes = rng.normal(size=len(FREQS)) ** 2
    ndofs = np.full(len(FREQS), 2)
    pars_init = guess(FREQS, ndofs, amplitudes, WEIGHTS, model, rng, 10)
    assert len(pars_init) == 3
    assert np.isfinite(pars_init).all()


def test_pade_npar():
    assert PadeModel([0, 2], [2]).npar == 3


PARS_REF_PADE = [
    [0.0, 0.0, 0.0, 0.0],
    [0.5, -2.0, 0.4, -8.3],
    [1.3, 2.0, -0.4, 0.0],
    [0.0, 0.5, 3.2, 1.3],
    [0.2, 0.9, 0.1, 0.0],
]


@pytest.mark.parametrize("model", [PadeModel([0, 1, 2], [2]), PadeModel([0, 2], [1, 2])])
def test_vectorize_pade_compute(model):
    check_vectorize_compute(model, PARS_REF_PADE)


@pytest.mark.parametrize("model", [PadeModel([0, 1, 2], [2]), PadeModel([0, 2], [1, 2])])
def test_vectorize_pade_prior(model):
    check_vectorize_prior(model, PARS_REF_PADE)


@pytest.mark.parametrize("pars_ref", PARS_REF_PADE)
def test_gradient_pade(pars_ref):
    pars_ref = np.array(pars_ref)
    model = PadeModel([0, 1, 2], [2])
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    check_gradient(lambda pars, *, deriv=0: model.compute(FREQS, pars, deriv=deriv), pars_ref)


@pytest.mark.parametrize("pars_ref", PARS_REF_PADE)
def test_hessian_pade(pars_ref):
    pars_ref = np.array(pars_ref)
    model = PadeModel([0, 1, 2], [2])
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    check_hessian(lambda pars, *, deriv=0: model.compute(FREQS, pars, deriv=deriv), pars_ref)


def test_guess_pade():
    model = PadeModel([0, 2], [2])
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    rng = np.random.default_rng(123)
    amplitudes = rng.normal(size=len(FREQS)) ** 2
    ndofs = np.full(len(FREQS), 2)
    pars_init = guess(FREQS, ndofs, amplitudes, WEIGHTS, model, rng, 10)
    assert len(pars_init) == 3
    assert np.isfinite(pars_init).all()


def test_guess_pade_detailed():
    freqs = np.linspace(0, 1.0, NFREQ)
    model = PadeModel([0, 2], [2])
    model.configure_scales(TIMESTEP, freqs, AMPLITUDES_REF)
    pars_ref = np.array([3.0, 1.5, 2.0])
    amplitudes_ref = model.compute(freqs, pars_ref, deriv=0)[0]
    x = freqs / freqs[-1]
    assert amplitudes_ref == pytest.approx((3.0 + 1.5 * x**2) / (1.0 + 2.0 * x**2), rel=1e-10)
    ndofs = np.full(len(freqs), 20)
    pars_init_low, amplitudes_low = model.solve_linear(freqs, ndofs, amplitudes_ref, WEIGHTS, [])
    assert pars_init_low == pytest.approx(pars_ref, rel=1e-10)
    assert amplitudes_low == pytest.approx(amplitudes_ref, rel=1e-10)
    rng = np.random.default_rng(123)
    pars_init = guess(freqs, ndofs, amplitudes_ref, WEIGHTS, model, rng, 10)
    assert pars_ref == pytest.approx(pars_init, rel=1e-10)
