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
"""Unit tests for ``stacie.cutoff``."""

import numpy as np
import pytest

from stacie.cutoff import (
    evidence_criterion,
    expected_ufc,
    general_ufc,
    halfapprox_criterion,
    halfhalf_criterion,
)
from stacie.utils import robust_posinv


def test_ufc_expectation_value():
    rng = np.random.default_rng(42)
    nfreq = 30
    basis = rng.standard_normal((4, nfreq))
    exp = expected_ufc(basis)

    ufcs = []
    for _ in range(10000):
        data = rng.standard_normal(nfreq)
        data -= np.linalg.lstsq(basis.T, data, rcond=None)[0] @ basis
        ufc = general_ufc(data)
        ufcs.append(ufc)
    assert exp == pytest.approx(np.mean(ufcs), rel=1e-1)


def test_halfhalf_preconditioned():
    npar = 4
    rng = np.random.default_rng(42)
    basis1 = rng.standard_normal((npar, npar))
    basis2 = rng.standard_normal((npar, npar))
    props = {
        "cost_hess_rescaled_evals": np.ones(npar),
        "cost_hess_half1": np.dot(basis1, basis1.T),
        "cost_hess_half2": np.dot(basis2, basis2.T),
        "cost_hess_scales": rng.uniform(2, 5, npar),
        "pars_half1": rng.standard_normal(npar),
        "pars_half2": rng.standard_normal(npar),
    }
    result1 = halfhalf_criterion(props)
    result2 = halfhalf_criterion(props, precondition=False)
    assert result1["criterion"] == pytest.approx(result2["criterion"], rel=1e-5)
    assert result1["criterion_expected"] == pytest.approx(result2["criterion_expected"], rel=1e-5)


@pytest.mark.parametrize("convergence_check", [True, False])
def test_halfapprox_preconditioned(convergence_check):
    npoint = 40
    npar = 4
    rng = np.random.default_rng(42)
    props = {
        "amplitudes": rng.uniform(3, 5, npoint),
        "amplitudes_model": [
            rng.uniform(3, 5, npoint),
            rng.uniform(3, 5, (npar, npoint)),
        ],
        "thetas": rng.uniform(3, 5, npoint),
        "kappas": np.full(npoint, 10),
    }
    result1 = halfapprox_criterion(props, convergence_check=convergence_check)
    result2 = halfapprox_criterion(props, convergence_check=convergence_check, precondition=False)
    assert result1["criterion"] == pytest.approx(result2["criterion"], rel=1e-5)
    assert result1["criterion_expected"] == pytest.approx(result2["criterion_expected"], rel=1e-5)


def test_evidence_criterion_scales():
    npar = 4
    rng = np.random.default_rng(42)
    basis = rng.standard_normal((npar, npar))
    hess = np.dot(basis, basis.T)
    evals1 = np.linalg.eigvalsh(hess)
    scales2, evals2 = robust_posinv(hess)[:2]
    result1 = evidence_criterion(
        {
            "ll": -5,
            "cost_hess_rescaled_evals": evals1,
            "cost_hess_scales": np.ones(npar),
        }
    )
    result2 = evidence_criterion(
        {
            "ll": -5,
            "cost_hess_rescaled_evals": evals2,
            "cost_hess_scales": scales2,
        }
    )
    assert result1["criterion"] == pytest.approx(result2["criterion"], rel=1e-5)
