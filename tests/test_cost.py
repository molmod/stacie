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
"""Tests for ``stacie.cost``."""

import numpy as np
import pytest
from conftest import check_curv, check_deriv, check_gradient, check_hessian
from scipy import stats

from stacie.cost import LowFreqCost, entropy_gamma, logpdf_gamma, varlogp_gamma
from stacie.model import PadeModel

LOGPDF_GAMMA_CASES = [
    (1.0, 1.0, 2.0),
    [1.2, 0.9, 2.2],
    [0.3, 0.001, 0.7],
    ([0.4, 1.1, 1.5], [1.0, 2.0, 3.0], [1.2, 4.3, 8.1]),
]


@pytest.mark.parametrize(("x", "alpha", "theta_ref"), LOGPDF_GAMMA_CASES)
def test_logpdf_gamma_deriv1(x, alpha, theta_ref):
    check_deriv(lambda theta, *, deriv=0: logpdf_gamma(x, alpha, theta, deriv=deriv), theta_ref)


@pytest.mark.parametrize(("x", "alpha", "theta_ref"), LOGPDF_GAMMA_CASES)
def test_logpdf_gamma_deriv2(x, alpha, theta_ref):
    check_curv(lambda theta, *, deriv=0: logpdf_gamma(x, alpha, theta, deriv=deriv), theta_ref)


@pytest.mark.parametrize(("x", "alpha", "theta_ref"), LOGPDF_GAMMA_CASES)
def test_entropy_gamma_deriv1(x, alpha, theta_ref):
    check_deriv(lambda theta, *, deriv=0: entropy_gamma(alpha, theta, deriv=deriv), theta_ref)


@pytest.mark.parametrize(("x", "alpha", "theta_ref"), LOGPDF_GAMMA_CASES)
def test_entropy_gamma_deriv2(x, alpha, theta_ref):
    check_curv(lambda theta, *, deriv=0: entropy_gamma(alpha, theta, deriv=deriv), theta_ref)


def test_entropy():
    """Check that the entropy matches the expectation value of -log(p)."""
    alpha = 2.5
    theta = 6.0
    entropy = entropy_gamma(alpha, theta)[0]
    rng = np.random.default_rng(1234)
    x = stats.gamma.rvs(alpha, scale=theta, size=10000, random_state=rng)
    logpdf1 = stats.gamma.logpdf(x, alpha, scale=theta)
    logpdf2 = logpdf_gamma(x, alpha, theta)[0]
    assert logpdf1 == pytest.approx(logpdf2, rel=1e-8)
    check = -np.mean(logpdf1)
    assert entropy == pytest.approx(check, rel=1e-2)


def test_varlogp_gamma():
    """Check the variance of the log PDF of the gamma distribution."""
    alpha = 2.5
    theta = 6.0
    varlogp = varlogp_gamma(alpha)
    rng = np.random.default_rng(1234)
    x = stats.gamma.rvs(alpha, scale=theta, size=100000, random_state=rng)
    logpdf = stats.gamma.logpdf(x, alpha, scale=theta)
    assert varlogp == pytest.approx(np.var(logpdf), rel=1e-2)


@pytest.fixture
def mycost():
    freqs = np.linspace(0, 4.0, 10)
    model = PadeModel([0, 2], [2])
    model.configure_scales(1.0, freqs, np.ones_like(freqs))
    amplitudes = np.array([1.5, 1.4, 1.1, 0.9, 0.8, 1.0, 0.9, 0.9, 0.8, 1.1])
    ndofs = np.array([5, 10, 10, 10, 10, 10, 10, 10, 10, 5])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    return LowFreqCost(freqs, ndofs, amplitudes, weights, model)


PARS_REF_PADE = [
    [1.2, 0.9, 2.2],
    [3.0, 0.5, 2.5],
    [0.1, 4.0, 2.7],
    [108.0, 77.7, 1.8],
]


def test_vectorize_pade(mycost):
    """Check that the model is vectorized."""
    results = mycost(PARS_REF_PADE, deriv=2)
    assert len(results) == 3
    nvec = len(PARS_REF_PADE)
    assert results[0].shape == (nvec,)
    assert results[1].shape == (nvec, 3)
    assert results[2].shape == (nvec, 3, 3)
    for i, one_pars_ref in enumerate(PARS_REF_PADE):
        one_results = mycost(one_pars_ref, deriv=2)
        assert one_results[0].shape == ()
        assert one_results[1].shape == (3,)
        assert one_results[2].shape == (3, 3)
        assert (one_results[0] == results[0][i]).all()
        assert (one_results[1] == results[1][i]).all()
        assert (one_results[2] == results[2][i]).all()


@pytest.mark.parametrize("pars_ref", PARS_REF_PADE)
def test_gradient_pade(mycost, pars_ref):
    check_gradient(mycost, pars_ref)


@pytest.mark.parametrize("pars_ref", PARS_REF_PADE)
def test_hessian_pade(mycost, pars_ref):
    check_hessian(mycost, pars_ref)
