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

from collections.abc import Callable

import attrs
import numdifftools as nd
import numpy as np
import pytest
from conftest import check_gradient, check_hessian

from stacie.model import (
    DebyeModel,
    ExpPolyModel,
    LorentzModel,
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


# The Lorentz and Debye models are both Pade(0,2;2) models with a reparametrization
# (offset, amplitude, correlation time).
# The cases below provide everything that differs between them,
# so that the tests that follow can be shared.


@attrs.define(frozen=True)
class PeakCase:
    """Everything the shared tests need to know about one of the Pade(0,2;2) special cases."""

    label: str
    """Name of the model, also used as suffix of ``pars_{label}`` in the derived properties."""

    model_class: type
    """The model class, whose constructor accepts ``ratio_weight`` and ``ratio_threshold``."""

    convert: Callable
    """The conversion function from Pade(0,2;2) parameters to the model's own parameters."""

    to_pade: Callable
    """Construct the Pade(0,2;2) parameters from the model's own parameters."""

    spectrum: Callable
    """Evaluate the spectrum in its textbook form, given frequencies and the model's parameters."""

    pars_pade_valid: list[float]
    """Pade(0,2;2) parameters accepted by the model."""

    pars_pade_invalid: list[tuple[list[float], str]]
    """Pade(0,2;2) parameters rejected by the model, each with the reason for the rejection."""


def lorentz_to_pade022(pars_lorentz):
    """Construct Pade(0,2;2) parameters from Lorentz parameters.

    Matching ``(p0 + p2 f^2) / (1 + q2 f^2)`` with
    ``C_0 + 2 C_1 tau / (1 + x^2)``, where ``x = 2 pi f tau``, gives
    ``q2 = (2 pi tau)^2``, ``p2 = C_0 q2`` (the limit of large ``f``)
    and ``p0 = C_0 + 2 C_1 tau`` (the value at ``f = 0``).
    """
    c0, c1, tau = pars_lorentz
    q2 = (2 * np.pi * tau) ** 2
    return np.array([c0 + 2 * c1 * tau, c0 * q2, q2])


def lorentz_spectrum(freqs, pars_lorentz):
    """Evaluate the Lorentz spectrum in its textbook form."""
    c0, c1, tau = pars_lorentz
    x2 = (2 * np.pi * freqs * tau) ** 2
    return c0 + 2 * c1 * tau / (1 + x2)


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


LORENTZ = PeakCase(
    label="lorentz",
    model_class=LorentzModel,
    convert=convert_pade022_lorentz,
    to_pade=lorentz_to_pade022,
    spectrum=lorentz_spectrum,
    # A valid Lorentz spectrum has p0 q2 > p2.
    pars_pade_valid=[2.0, 0.5, 1.0],
    pars_pade_invalid=[
        ([1.0, 0.5, 0.2], "minimum at zero"),  # p0 q2 < p2.
        ([1.0, 1.0, 1.0], "flat"),  # No peak at all: p0 q2 == p2.
        ([1.0, -5.0, -1.0], "q2 < 0"),
        ([1.0, -5.0, 0.0], "q2 == 0"),
        ([-1.0, -5.0, 1.0], "p0 < 0"),
        ([0.0, -1.0, 1.0], "p0 == 0"),
    ],
)

DEBYE = PeakCase(
    label="debye",
    model_class=DebyeModel,
    convert=convert_pade022_debye,
    to_pade=debye_to_pade022,
    spectrum=debye_spectrum,
    # A valid Debye spectrum has p2 > p0 q2.
    pars_pade_valid=[1.0, 0.5, 0.2],
    pars_pade_invalid=[
        ([2.0, 0.5, 1.0], "maximum at zero"),  # p2 < p0 q2.
        ([1.0, 1.0, 1.0], "flat"),  # No dip at all: p2 == p0 q2.
        ([1.0, 0.5, -1.0], "q2 < 0"),
        ([1.0, 0.5, 0.0], "q2 == 0"),
        ([-1.0, 1.0, 1.0], "p0 < 0"),
        ([0.0, 1.0, 1.0], "p0 == 0"),
    ],
)

CASES = [LORENTZ, DEBYE]
parametrize_case = pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)

# Model parameters (offset, amplitude, tau_exp), valid for both the Lorentz and the Debye model.
PARS_REF_PEAK = [
    [1.0, 2.0, 0.3],
    [0.5, 0.1, 1.7],
    [3.0, 0.7, 0.05],
]


def tau_exp_var(pars_pade, covar_pade):
    """Variance of tau = sqrt(q2) / (2 pi), which only depends on q2 in the Pade parameters.

    Linear error propagation gives ``var(tau) = var(q2) / (16 pi^2 q2)``.
    """
    return covar_pade[2, 2] / (16 * np.pi**2 * pars_pade[2])


@parametrize_case
def test_peak_properties(case):
    model = case.model_class()
    assert model.npar == 3
    assert model.name == f"{case.label}()"
    assert list(model.numer_degrees) == [0, 2]
    assert list(model.denom_degrees) == [2]


@parametrize_case
@pytest.mark.parametrize("pars", PARS_REF_PEAK)
def test_peak_compute(case, pars):
    model = case.model_class()
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    pars_pade = case.to_pade(pars)
    amplitudes = model.compute(FREQS, pars_pade, deriv=2)
    # Agreement with the textbook form of the spectrum.
    assert amplitudes[0] == pytest.approx(case.spectrum(FREQS, pars), rel=1e-10)
    # The model is a Pade model with fixed degrees.
    amplitudes_pade = PadeModel([0, 2], [2]).compute(FREQS, pars_pade, deriv=2)
    for amplitudes_deriv, amplitudes_pade_deriv in zip(amplitudes, amplitudes_pade, strict=True):
        assert amplitudes_deriv == pytest.approx(amplitudes_pade_deriv, rel=1e-10)


@parametrize_case
@pytest.mark.parametrize("pars", PARS_REF_PEAK)
def test_convert_pade022_roundtrip(case, pars):
    pars_conv, _ = case.convert(case.to_pade(pars), np.eye(3))
    assert pars_conv == pytest.approx(pars, rel=1e-10)


@parametrize_case
@pytest.mark.parametrize("pars", PARS_REF_PEAK)
def test_convert_pade022_covar(case, pars):
    pars_pade = case.to_pade(pars)
    covar_pade = 1e-3 * np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])
    _, covar_conv = case.convert(pars_pade, covar_pade)
    assert covar_conv.shape == (3, 3)
    assert covar_conv == pytest.approx(covar_conv.T, abs=1e-13)
    assert covar_conv[2, 2] == pytest.approx(tau_exp_var(pars_pade, covar_pade), rel=1e-10)

    # Numerical check of the full covariance via finite differences.
    def transform(pars):
        return case.convert(pars, covar_pade)[0]

    num_jacobian = nd.Jacobian(transform)(pars_pade)
    covar_conv_fd = num_jacobian @ covar_pade @ num_jacobian.T
    assert covar_conv == pytest.approx(covar_conv_fd, rel=1e-6)


@pytest.mark.parametrize("pars", PARS_REF_PEAK)
def test_convert_pade022_debye_dc_variance(pars):
    # The DC level is p0, so its variance is untouched.
    covar_pade = 1e-3 * np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])
    _, covar_conv = convert_pade022_debye(debye_to_pade022(pars), covar_pade)
    assert covar_conv[0, 0] == pytest.approx(covar_pade[0, 0], rel=1e-10)


@parametrize_case
@pytest.mark.parametrize("pars", PARS_REF_PEAK)
def test_peak_solve_linear_exact(case, pars):
    # A noiseless spectrum is reproduced exactly by the linear Pade fit.
    freqs = np.linspace(0, 1.0, NFREQ)
    model = case.model_class()
    model.configure_scales(TIMESTEP, freqs, AMPLITUDES_REF)
    amplitudes_ref = case.spectrum(freqs, pars)
    ndofs = np.full(len(freqs), 20)
    pars_pade, amplitudes = model.solve_linear(freqs, ndofs, amplitudes_ref, WEIGHTS, [])
    assert amplitudes == pytest.approx(amplitudes_ref, rel=1e-10)
    pars_conv, _ = case.convert(pars_pade, np.eye(3))
    assert pars_conv == pytest.approx(pars, rel=1e-8)


@parametrize_case
def test_peak_guess(case):
    model = case.model_class()
    model.configure_scales(TIMESTEP, FREQS, AMPLITUDES_REF)
    rng = np.random.default_rng(123)
    amplitudes = rng.normal(size=len(FREQS)) ** 2
    ndofs = np.full(len(FREQS), 2)
    pars_init = guess(FREQS, ndofs, amplitudes, WEIGHTS, model, rng, 10)
    assert len(pars_init) == 3
    assert np.isfinite(pars_init).all()


def make_peak_props(pars_pade, covar_diag=(1e-4, 1e-4, 1e-4), **kwargs):
    return {"pars": np.array(pars_pade), "pars_covar": np.diag(covar_diag), **kwargs}


@parametrize_case
@pytest.mark.parametrize("with_criterion", [False, True])
def test_peak_derive_props(case, with_criterion):
    pars = np.array([1.0, 2.0, 0.3])
    pars_pade = case.to_pade(pars)
    props = make_peak_props(pars_pade, **({"criterion": 5.0} if with_criterion else {}))
    covar_pade = props["pars_covar"].copy()
    model = case.model_class()
    model.derive_props(props)

    # The autocorrelation integral is the spectrum at zero frequency, which is p0.
    acint = case.spectrum(0.0, pars)
    assert props["acint"] == pytest.approx(acint, rel=1e-10)
    assert props["acint_var"] == pytest.approx(covar_pade[0, 0], rel=1e-10)

    # Model parameters and the relaxation time.
    assert props[f"pars_{case.label}"] == pytest.approx(pars, rel=1e-10)
    assert props[f"pars_{case.label}_covar"].shape == (3, 3)
    assert props["corrtime_exp"] == pytest.approx(pars[2], rel=1e-10)
    tau_var = tau_exp_var(pars_pade, covar_pade)
    assert props["corrtime_exp_var"] == pytest.approx(tau_var, rel=1e-10)

    # Derived times are linear in tau, so their variances scale with the squared prefactor.
    assert props["exp_block_time"] == pytest.approx(np.pi / 10 * pars[2], rel=1e-10)
    assert props["exp_block_time_var"] == pytest.approx((np.pi / 10) ** 2 * tau_var, rel=1e-10)
    assert props["exp_simulation_time"] == pytest.approx(20 * np.pi * pars[2], rel=1e-10)
    assert props["exp_simulation_time_var"] == pytest.approx((20 * np.pi) ** 2 * tau_var, rel=1e-10)

    if with_criterion:
        # The penalty is ratio_weight times the ratio of relative errors.
        relerr_tau = np.sqrt(tau_var) / pars[2]
        relerr_acint = np.sqrt(covar_pade[0, 0]) / acint
        assert props["criterion"] == pytest.approx(
            5.0 + model.ratio_weight * relerr_tau / relerr_acint, rel=1e-10
        )
        assert "msg" not in props
    else:
        assert "criterion" not in props


@pytest.mark.parametrize(
    ("case", "pars_pade", "reason"),
    [
        pytest.param(case, pars_pade, reason, id=f"{case.label}-{reason or 'valid'}")
        for case in CASES
        for pars_pade, reason in [(case.pars_pade_valid, None), *case.pars_pade_invalid]
    ],
)
def test_peak_derive_props_validity(case, pars_pade, reason):
    props = make_peak_props(pars_pade, criterion=1.0)
    case.model_class().derive_props(props)
    # The autocorrelation integral is always derived.
    assert props["acint"] == pars_pade[0]
    if reason is None:
        assert np.isfinite(props["criterion"])
        assert f"pars_{case.label}" in props
    else:
        assert props["criterion"] == np.inf
        assert props["msg"] == "No correlation time estimate available."
        assert f"pars_{case.label}" not in props
        assert "corrtime_exp" not in props


@parametrize_case
@pytest.mark.parametrize(
    ("ratio_threshold", "expect_discard"),
    [(100.0, False), (1e-3, True)],
)
def test_peak_derive_props_threshold(case, ratio_threshold, expect_discard):
    pars_pade = case.to_pade([1.0, 2.0, 0.3])
    props = make_peak_props(pars_pade, criterion=1.0)
    case.model_class(ratio_threshold=ratio_threshold).derive_props(props)
    if expect_discard:
        assert props["criterion"] == np.inf
        assert props["msg"].startswith("rel.err. tau_exp >")
    else:
        assert np.isfinite(props["criterion"])
        assert "msg" not in props


@parametrize_case
@pytest.mark.parametrize("ratio_weight", [0.0, 1.0, 2.5])
def test_peak_derive_props_ratio_weight(case, ratio_weight):
    pars_pade = case.to_pade([1.0, 2.0, 0.3])
    props = make_peak_props(pars_pade, criterion=1.0)
    case.model_class(ratio_weight=ratio_weight).derive_props(props)
    props_ref = make_peak_props(pars_pade, criterion=1.0)
    case.model_class(ratio_weight=1.0).derive_props(props_ref)
    ratio = props_ref["criterion"] - 1.0
    assert props["criterion"] == pytest.approx(1.0 + ratio_weight * ratio, rel=1e-10)
