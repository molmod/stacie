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

import pytest
from path import Path
from stacie.estimate import estimate_acfint
from stacie.plot import plot
from stacie.spectrum import Spectrum
from stacie.zarr import load

DOUBLE_NAMES = ["double1", "double2"]
WHITE_NAMES = ["white1", "white2"]
NAME_LISTS = DOUBLE_NAMES, WHITE_NAMES
ALL_NAMES = [j for i in NAME_LISTS for j in i]


def plot_test_result(prefix, res):
    dn_out = Path("tests/outputs")
    dn_out.makedirs_p()
    path_pdf = dn_out / f"{prefix}.pdf"
    plot(path_pdf, res)


def register_result(regtest, res):
    with regtest:
        print(f"ACF Int  = {res.props['acfint']:.4f} ± {res.props['acfint_std']:.4f}")
        print(f"Tau tail = {res.props['corrtime_tail']:.4f} ± {res.props['corrtime_tail_std']:.4f}")
        print(f"Log lh   = {res.props['ll']:.5e}")
        print("---")


def check_noscan_single(regtest, spectrum, prefix, fcut=0.005):
    res = estimate_acfint(spectrum, fcut=fcut, maxscan=1)
    register_result(regtest, res)
    plot_test_result(prefix, res)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_noscan(regtest, name):
    spectrum = load(f"tests/inputs/spectrum_{name}.zip", Spectrum)
    check_noscan_single(regtest, spectrum, f"noscan_{name}")


@pytest.mark.parametrize(("name", "fcut"), [("white2", 0.008), ("double1", 0.05)])
def test_noscan_fail(regtest, name, fcut):
    spectrum = load(f"tests/inputs/spectrum_{name}.zip", Spectrum)
    check_noscan_single(regtest, spectrum, f"noscan_{name}_fail", fcut=fcut)


@pytest.mark.parametrize("names", NAME_LISTS)
def test_noscan_multi(regtest, names):
    res = []
    for name in names:
        spectrum = load(f"tests/inputs/spectrum_{name}.zip", Spectrum)
        r = estimate_acfint(spectrum, fcut=0.005, maxscan=1)
        register_result(regtest, r)
        res.append(r)
    plot_test_result("noscan_multi", res)


def check_scan_single(regtest, spectrum, prefix):
    res = estimate_acfint(spectrum, fcut=0.01, maxscan=10)
    register_result(regtest, res)
    plot_test_result(prefix, res)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_scan(regtest, name):
    spectrum = load(f"tests/inputs/spectrum_{name}.zip", Spectrum)
    check_scan_single(regtest, spectrum, f"scan_{name}")


@pytest.mark.parametrize("names", NAME_LISTS)
def test_scan_multi(regtest, names):
    res = []
    for name in names:
        spectrum = load(f"tests/inputs/spectrum_{name}.zip", Spectrum)
        r = estimate_acfint(spectrum, fcut=0.01, maxscan=10)
        register_result(regtest, r)
        res.append(r)
    plot_test_result("scan_multi", res)
