# STACIE is a STable AutoCorrelation Integral Estimator.
# Copyright 2024-2026 The contributors of the STACIE Python Package.
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

import numdifftools as nd
import numpy as np
import pytest
from conftest import check_gradient, check_hessian

from stacie.model import (
    DebyeModel,
    ExpPolyModel,
    PadeModel,
    convert_pade022_debye,
    convert_pade022_lorentz,
    guess,
)

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


def test_convert_pade022_lorentz():
    pars_pade = np.array([2.0, 3.0, 4.0])
    covar_pade = np.array([[0.1, 0.01, 0.02], [0.01, 0.2, 0.03], [0.02, 0.03, 0.3]])
    pars_lorentz, covar_lorentz = convert_pade022_lorentz(pars_pade, covar_pade)
    assert pars_lorentz.shape == (3,)
    assert covar_lorentz.shape == (3, 3)
    assert covar_lorentz == pytest.approx(covar_lorentz.T, abs=1e-13)

    # Numerical check via finite differences
    def transform(pars):
        return convert_pade022_lorentz(pars, covar_pade)[0]

    num_jacobian = nd.Jacobian(transform)(pars_pade)
    covar_lorentz_fd = num_jacobian @ covar_pade @ num_jacobian.T
    assert covar_lorentz == pytest.approx(covar_lorentz_fd, rel=1e-6)


# Debye parameters (C_dc, C_delta, tau_exp) with a minimum at zero frequency.
PARS_REF_DEBYE = [
    [1.0, 2.0, 0.3],
    [0.5, 0.1, 1.7],
    [3.0, 0.7, 0.05],
]


def debye_to_pade022(pars_debye):
    """Construct Pade(0,2;2) parameters from Debye parameters.

    Matching ``(p0 + p2 f^2) / (1 + q2 f^2)`` with
    ``C_dc + C_delta x^2 / (1 + x^2)``, where ``x = 2 pi f tau``, gives
    ``q2 = (2 pi tau)^2``, ``p0 = C_dc`` and ``p2 = (C_dc + C_delta) q2``.
    """
    c_dc, c_delta, tau = pars_debye
    q2 = (2 * np.pi * tau) ** 2
    return np.array([c_dc, (c_dc + c_delta) * q2, q2])


def debye_spectrum(freqs, pars_debye):
    """Evaluate the Debye spectrum in its textbook form."""
    c_dc, c_delta, tau = pars_debye
    x2 = (2 * np.pi * freqs * tau) ** 2
    return c_dc + c_delta * x2 / (1 + x2)


def test_debye_properties():
    model = DebyeModel()
    assert model.npar == 3
    assert model.name == "debye()"
    assert list(model.numer_degrees) == [0, 2]
    assert list(model.denom_degrees) == [2]


@pytest.mark.parametrize("pars_debye", PARS_REF_DEBYE)
def test_debye_compute(pars_debye):
    model = DebyeModel()
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    pars_pade = debye_to_pade022(pars_debye)
    amplitudes = model.compute(FREQS, pars_pade, deriv=2)
    # Agreement with the textbook form of the Debye spectrum.
    assert amplitudes[0] == pytest.approx(debye_spectrum(FREQS, pars_debye), rel=1e-10)
    # The model is a Pade model with fixed degrees.
    amplitudes_pade = PadeModel([0, 2], [2]).compute(FREQS, pars_pade, deriv=2)
    for amplitudes_debye_deriv, amplitudes_pade_deriv in zip(
        amplitudes, amplitudes_pade, strict=True
    ):
        assert amplitudes_debye_deriv == pytest.approx(amplitudes_pade_deriv, rel=1e-10)


@pytest.mark.parametrize("pars_debye", PARS_REF_DEBYE)
def test_convert_pade022_debye_roundtrip(pars_debye):
    pars_pade = debye_to_pade022(pars_debye)
    pars_debye_conv, _ = convert_pade022_debye(pars_pade, np.eye(3))
    assert pars_debye_conv == pytest.approx(pars_debye, rel=1e-10)


@pytest.mark.parametrize("pars_debye", PARS_REF_DEBYE)
def test_convert_pade022_debye_covar(pars_debye):
    pars_pade = debye_to_pade022(pars_debye)
    covar_pade = 1e-3 * np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])
    _, covar_conv = convert_pade022_debye(pars_pade, covar_pade)
    assert covar_conv.shape == (3, 3)
    assert covar_conv == pytest.approx(covar_conv.T, abs=1e-13)

    # The DC level is p0, so its variance is untouched.
    assert covar_conv[0, 0] == pytest.approx(covar_pade[0, 0], rel=1e-10)

    # The relaxation time only depends on q2: tau = sqrt(q2) / (2 pi),
    # so var(tau) = var(q2) / (16 pi^2 q2).
    assert covar_conv[2, 2] == pytest.approx(
        covar_pade[2, 2] / (16 * np.pi**2 * pars_pade[2]), rel=1e-10
    )

    # Numerical check of the full covariance via finite differences.
    def transform(pars):
        return convert_pade022_debye(pars, covar_pade)[0]

    num_jacobian = nd.Jacobian(transform)(pars_pade)
    covar_conv_fd = num_jacobian @ covar_pade @ num_jacobian.T
    assert covar_conv == pytest.approx(covar_conv_fd, rel=1e-6)


@pytest.mark.parametrize("pars_debye", PARS_REF_DEBYE)
def test_debye_solve_linear_exact(pars_debye):
    # A noiseless Debye spectrum is reproduced exactly by the linear Pade fit.
    freqs = np.linspace(0, 1.0, NFREQ)
    model = DebyeModel()
    model.configure_scales(TIMESTEP, freqs, AMPLITUDES_REF)
    amplitudes_ref = debye_spectrum(freqs, pars_debye)
    ndofs = np.full(len(freqs), 20)
    pars_pade, amplitudes = model.solve_linear(freqs, ndofs, amplitudes_ref, WEIGHTS, [])
    assert amplitudes == pytest.approx(amplitudes_ref, rel=1e-10)
    pars_debye_conv, _ = convert_pade022_debye(pars_pade, np.eye(3))
    assert pars_debye_conv == pytest.approx(pars_debye, rel=1e-8)


def test_guess_debye():
    model = DebyeModel()
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    rng = np.random.default_rng(123)
    amplitudes = rng.normal(size=len(FREQS)) ** 2
    ndofs = np.full(len(FREQS), 2)
    pars_init = guess(FREQS, ndofs, amplitudes, WEIGHTS, model, rng, 10)
    assert len(pars_init) == 3
    assert np.isfinite(pars_init).all()


def make_debye_props(pars_pade, covar_diag=(1e-4, 1e-4, 1e-4), **kwargs):
    return {"pars": np.array(pars_pade), "pars_covar": np.diag(covar_diag), **kwargs}


@pytest.mark.parametrize("with_criterion", [False, True])
def test_debye_derive_props(with_criterion):
    pars_debye = np.array([1.0, 2.0, 0.3])
    pars_pade = debye_to_pade022(pars_debye)
    props = make_debye_props(pars_pade, **({"criterion": 5.0} if with_criterion else {}))
    covar_pade = props["pars_covar"].copy()
    model = DebyeModel()
    model.derive_props(props)

    # The autocorrelation integral is the DC level.
    assert props["acint"] == pytest.approx(pars_debye[0], rel=1e-10)
    assert props["acint_var"] == pytest.approx(covar_pade[0, 0], rel=1e-10)

    # Debye parameters and the relaxation time.
    assert props["pars_debye"] == pytest.approx(pars_debye, rel=1e-10)
    assert props["pars_debye_covar"].shape == (3, 3)
    assert props["corrtime_exp"] == pytest.approx(pars_debye[2], rel=1e-10)
    tau_var = covar_pade[2, 2] / (16 * np.pi**2 * pars_pade[2])
    assert props["corrtime_exp_var"] == pytest.approx(tau_var, rel=1e-10)

    # Derived times are linear in tau, so their variances scale with the squared prefactor.
    assert props["exp_block_time"] == pytest.approx(np.pi / 10 * pars_debye[2], rel=1e-10)
    assert props["exp_block_time_var"] == pytest.approx((np.pi / 10) ** 2 * tau_var, rel=1e-10)
    assert props["exp_simulation_time"] == pytest.approx(20 * np.pi * pars_debye[2], rel=1e-10)
    assert props["exp_simulation_time_var"] == pytest.approx((20 * np.pi) ** 2 * tau_var, rel=1e-10)

    if with_criterion:
        # The penalty is ratio_weight times the ratio of relative errors.
        relerr_tau = np.sqrt(tau_var) / pars_debye[2]
        relerr_acint = np.sqrt(covar_pade[0, 0]) / pars_debye[0]
        assert props["criterion"] == pytest.approx(
            5.0 + model.ratio_weight * relerr_tau / relerr_acint, rel=1e-10
        )
        assert "msg" not in props
    else:
        assert "criterion" not in props


@pytest.mark.parametrize(
    ("pars_pade", "reason"),
    [
        ([1.0, 0.5, 0.2], None),  # Control: a valid Debye spectrum (p2 > p0 q2).
        ([2.0, 0.5, 1.0], "maximum at zero"),  # Lorentz-like: p2 < p0 q2.
        ([1.0, 1.0, 1.0], "flat"),  # No dip at all: p2 == p0 q2.
        ([1.0, 0.5, -1.0], "q2 < 0"),
        ([1.0, 0.5, 0.0], "q2 == 0"),
        ([-1.0, 1.0, 1.0], "p0 < 0"),
        ([0.0, 1.0, 1.0], "p0 == 0"),
    ],
)
def test_debye_derive_props_validity(pars_pade, reason):
    props = make_debye_props(pars_pade, criterion=1.0)
    DebyeModel().derive_props(props)
    # The autocorrelation integral is always derived.
    assert props["acint"] == pars_pade[0]
    if reason is None:
        assert np.isfinite(props["criterion"])
        assert "pars_debye" in props
    else:
        assert props["criterion"] == np.inf
        assert props["msg"] == "No correlation time estimate available."
        assert "pars_debye" not in props
        assert "corrtime_exp" not in props


@pytest.mark.parametrize(
    ("ratio_threshold", "expect_discard"),
    [(100.0, False), (1e-3, True)],
)
def test_debye_derive_props_threshold(ratio_threshold, expect_discard):
    pars_pade = debye_to_pade022([1.0, 2.0, 0.3])
    props = make_debye_props(pars_pade, criterion=1.0)
    model = DebyeModel(ratio_threshold=ratio_threshold)
    model.derive_props(props)
    if expect_discard:
        assert props["criterion"] == np.inf
        assert props["msg"].startswith("rel.err. tau_exp >")
    else:
        assert np.isfinite(props["criterion"])
        assert "msg" not in props


@pytest.mark.parametrize("ratio_weight", [0.0, 1.0, 2.5])
def test_debye_derive_props_ratio_weight(ratio_weight):
    pars_pade = debye_to_pade022([1.0, 2.0, 0.3])
    props = make_debye_props(pars_pade, criterion=1.0)
    DebyeModel(ratio_weight=ratio_weight).derive_props(props)
    props_ref = make_debye_props(pars_pade, criterion=1.0)
    DebyeModel(ratio_weight=1.0).derive_props(props_ref)
    ratio = props_ref["criterion"] - 1.0
    assert props["criterion"] == pytest.approx(1.0 + ratio_weight * ratio, rel=1e-10)
