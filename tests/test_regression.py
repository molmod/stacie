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
"""Regression tests for typical Stacie workflows."""

import numpy as np
import pytest
from path import Path
from stacie.estimate import estimate_acfint
from stacie.plot import plot
from stacie.spectrum import Spectrum


@pytest.fixture()
def synth_30():
    amplitudes = np.load("tests/spectrum_30.npy")
    amplitudes_ref = np.load("tests/spectrum_ref.npy")
    ndof = np.full(amplitudes.shape, 800)
    ndof[0] = 400
    ndof[-1] = 400
    nstep = 2000
    freqs = np.fft.rfftfreq(nstep)
    return Spectrum(ndof, 0.5, np.arange(nstep), freqs, amplitudes, amplitudes_ref)


@pytest.fixture()
def synth_36():
    amplitudes = np.load("tests/spectrum_36.npy")
    amplitudes_ref = np.load("tests/spectrum_ref.npy")
    ndof = np.full(amplitudes.shape, 800)
    ndof[0] = 400
    ndof[-1] = 400
    nstep = 2000
    freqs = np.fft.rfftfreq(nstep)
    return Spectrum(ndof, 0.5, np.arange(nstep), freqs, amplitudes, amplitudes_ref)


def plot_test_result(prefix, res):
    dn_out = Path("tests/outputs")
    dn_out.makedirs_p()
    path_pdf = dn_out / f"{prefix}.pdf"
    plot(path_pdf, res)


def check_noscan_single(spectrum, regtest, prefix):
    res = estimate_acfint(spectrum, fcut=0.005, maxscan=1)
    with regtest:
        print(f"ACF Int  = {res.acfint:.4f} ± {res.acfint_std:.4f}")
        print(f"Tau tail = {res.corrtime_tail:.4f} ± {res.corrtime_tail_std:.4f}")
        print(f"Log lh   = {res.props['ll']:.5e}")
    plot_test_result(prefix, res)


def test_noscan_30(synth_30, regtest):
    check_noscan_single(synth_30, regtest, "noscan_30")


def test_noscan_36(synth_36, regtest):
    check_noscan_single(synth_36, regtest, "noscan_36")


def test_noscan_multi(synth_30, synth_36, regtest):
    res = []
    for spectrum in synth_30, synth_36:
        r = estimate_acfint(spectrum, fcut=0.005, maxscan=1)
        with regtest:
            print(f"ACF Int  = {r.acfint:.4f} ± {r.acfint_std:.4f}")
            print(f"Tau tail = {r.corrtime_tail:.4f} ± {r.corrtime_tail_std:.4f}")
            print(f"Log lh   = {r.props['ll']:.5e}")
            print("--")
        res.append(r)
    plot_test_result("noscan_multi", res)


def check_scan_single(spectrum, regtest, prefix):
    res = estimate_acfint(spectrum, fcut=0.01, maxscan=10)
    with regtest:
        print(f"ACF Int  = {res.acfint:.4f} ± {res.acfint_std:.4f}")
        print(f"Tau tail = {res.corrtime_tail:.4f} ± {res.corrtime_tail_std:.4f}")
        print(f"Log lh   = {res.props['ll']:.5e}")
    plot_test_result(prefix, res)


def test_scan_30(synth_30, regtest):
    check_scan_single(synth_30, regtest, "scan_30")


def test_scan_36(synth_36, regtest):
    check_scan_single(synth_36, regtest, "scan_36")


def test_scan_multi(synth_30, synth_36, regtest):
    res = []
    for spectrum in synth_30, synth_36:
        r = estimate_acfint(spectrum, fcut=0.01, maxscan=10)
        with regtest:
            print(f"ACF Int  = {r.acfint:.4f} ± {r.acfint_std:.4f}")
            print(f"Tau tail = {r.corrtime_tail:.4f} ± {r.corrtime_tail_std:.4f}")
            print(f"Log lh   = {r.props['ll']:.5e}")
            print("--")
        res.append(r)
    plot_test_result("scan_multi", res)
