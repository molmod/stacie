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
"""Unit tests for ``stacie.cutoff``."""

import numpy as np
import pytest

from stacie.cutoff import (
    CV2LCriterion,
    linear_weighted_regression,
)
from stacie.model import ExpPolyModel
from stacie.spectrum import compute_spectrum


def test_cv2l_preconditioned():
    nstep = 400
    rng = np.random.default_rng(42)
    spectrum = compute_spectrum(rng.standard_normal((10, nstep)))
    model = ExpPolyModel([0, 1])
    pars = np.array([0.1, 0.2])
    fcut = spectrum.freqs[nstep // 4]
    ncut = nstep // 3
    props = {
        "fcut": fcut,
        "ncut": ncut,
        "pars": pars,
        "switch_exponent": 20.0,
    }
    result1 = CV2LCriterion(regularize=False)(spectrum, model, props)
    result2 = CV2LCriterion(regularize=False, precondition=False)(spectrum, model, props)
    assert result1["criterion"] == pytest.approx(result2["criterion"], rel=1e-5)


def test_cv2l_regularize():
    nstep = 400
    rng = np.random.default_rng(42)
    spectrum = compute_spectrum(rng.standard_normal((10, nstep)))
    model = ExpPolyModel([0, 1])
    pars = np.array([0.1, 0.2])
    fcut = spectrum.freqs[nstep // 4]
    ncut = nstep // 3
    props = {
        "fcut": fcut,
        "ncut": ncut,
        "pars": pars,
        "switch_exponent": 20.0,
        "cost_hess_scales": np.array([1.0, 10.0]),
        "cost_hess_rescaled_evals": np.array([1.0, 2.0]),
        "cost_hess_rescaled_evecs": np.array([[1.0, 0.0], [0.0, 1.0]]),
    }
    result1 = CV2LCriterion()(spectrum, model, props)
    result2 = CV2LCriterion(precondition=False)(spectrum, model, props)
    assert result1["criterion"] == pytest.approx(result2["criterion"], rel=1e-5)


def test_linear_weighted_regression_lc():
    neq = 10
    npar = 4
    nw = 2
    rng = np.random.default_rng(42)
    dm = rng.standard_normal((neq, npar))
    ev = rng.standard_normal(neq)
    ws = rng.uniform(0, 1, (nw, neq))

    # Without linear combination
    xs1, cs1 = linear_weighted_regression(dm, ev, ws)
    assert xs1.shape == (nw, npar)
    assert cs1.shape == (nw, npar, nw, npar)
    assert cs1[0, :, 0] == pytest.approx(cs1[0, :, 0].T)
    assert cs1[1, :, 1] == pytest.approx(cs1[1, :, 1].T)
    assert cs1[1, :, 0] == pytest.approx(cs1[0, :, 1].T)
    xd = xs1[0] - xs1[1]
    cd = cs1[0, :, 0] + cs1[1, :, 1] - (cs1[0, :, 1] + cs1[1, :, 0])
    assert np.all(np.isfinite(xd))
    assert np.all(np.isfinite(cd))

    # With linear combination
    lc = np.array([[1.0, -1.0]])
    xs2, cs2 = linear_weighted_regression(dm, ev, ws, lc)
    assert xs2.shape == (1, npar)
    assert cs2.shape == (1, npar, 1, npar)
    assert xs2[0] == pytest.approx(xd)
    assert cs2[0, :, 0] == pytest.approx(cd)


def test_linear_weighted_regression_uncorrelated():
    neq = 10
    npar = 4
    nw = 2
    rng = np.random.default_rng(123)
    dm = rng.standard_normal((neq, npar))
    ev = rng.standard_normal(neq)
    ws = np.ones((nw, neq))
    ws[0, 5:] = 0
    ws[1, :5] = 0
    xs, cs = linear_weighted_regression(dm, ev, ws)
    assert xs[0] == pytest.approx(np.linalg.lstsq(dm[:5], ev[:5], rcond=None)[0], rel=1e-5)
    assert xs[1] == pytest.approx(np.linalg.lstsq(dm[5:], ev[5:], rcond=None)[0], rel=1e-5)
    assert cs[0, :, 1] == pytest.approx(np.zeros((npar, npar)), abs=1e-5)
    assert cs[1, :, 0] == pytest.approx(np.zeros((npar, npar)), abs=1e-5)


def test_linear_weighted_regression_example():
    neq = 50
    npar = 2
    nw = 2
    rng = np.random.default_rng(0)
    t = np.linspace(-1, 1, neq)
    dm = np.vstack((np.ones(neq), t)).T
    ref = 5 * t**2
    ws = np.array([(1 + t) / 2, (1 - t) / 2])

    all_xs = []
    cs_mean = 0
    nrep = 10000
    for _ in range(nrep):
        ev = rng.standard_normal(neq) + ref
        xs, cs = linear_weighted_regression(dm, ev, ws)
        all_xs.append(xs)
        cs_mean += cs
    all_xs = np.array(all_xs)
    cs_mean /= nrep
    assert all_xs.shape == (nrep, nw, npar)
    assert cs_mean[0, :, 0] == pytest.approx(np.cov(all_xs[:, 0, :], rowvar=False), abs=2e-3)
    assert cs_mean[1, :, 1] == pytest.approx(np.cov(all_xs[:, 1, :], rowvar=False), abs=2e-3)
    assert cs_mean.reshape((nw * npar, nw * npar)) == pytest.approx(
        np.cov(all_xs[:, 0, :], all_xs[:, 1, :], rowvar=False), abs=2e-3
    )
