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

from contextlib import ExitStack

import pytest
from path import Path
from stacie.estimate import FCutWarning, estimate_acfint
from stacie.model import ExpTailModel, SpectrumModel, WhiteNoiseModel
from stacie.msgpack import dump, load
from stacie.plot import plot
from stacie.spectrum import Spectrum

DOUBLE_NAMES = ["double1", "double2"]
WHITE_NAMES = ["white1", "white2"]
NAME_LISTS = DOUBLE_NAMES, WHITE_NAMES
ALL_NAMES = [j for i in NAME_LISTS for j in i]


def output_test_result(prefix, res):
    dn_out = Path("tests/outputs")
    dn_out.makedirs_p()
    path_pdf = dn_out / f"{prefix}.pdf"
    plot(path_pdf, res)
    path_zip = dn_out / f"{prefix}.nmpk.xz"
    dump(path_zip, res)


def register_result(regtest, res):
    with regtest:
        print(f"ACF Int  = {res.props['acfint']:.4e} ± {res.props['acfint_std']:.3e}")
        print(f"Tau tail = {res.props['corrtime_tail']:.4e} ± {res.props['corrtime_tail_std']:.3e}")
        print(f"Log lh   = {res.props['ll']:.5e}")
        print("---")


def check_noscan_single(
    regtest,
    spectrum: Spectrum,
    prefix: str,
    fcutmax: float = 0.005,
    model: SpectrumModel | None = None,
):
    res = estimate_acfint(spectrum, fcutmax=fcutmax, maxscan=1, model=model)
    register_result(regtest, res)
    output_test_result(prefix, res)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_epxtail_noscan(regtest, name):
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    check_noscan_single(regtest, spectrum, f"epxtail_noscan_{name}")


@pytest.mark.parametrize("name", WHITE_NAMES)
def test_white_noscan(regtest, name):
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    check_noscan_single(
        regtest, spectrum, f"white_noscan_{name}", fcutmax=0.1, model=WhiteNoiseModel()
    )


@pytest.mark.parametrize(("name", "fcutmax"), [("white2", 0.008), ("double1", 0.05)])
def test_exptail_noscan_fail(regtest, name, fcutmax):
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    check_noscan_single(regtest, spectrum, f"exptail_noscan_{name}_fail", fcutmax=fcutmax)


@pytest.mark.parametrize("names", NAME_LISTS)
def test_exptail_noscan_multi(regtest, names):
    res = []
    for name in names:
        spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
        r = estimate_acfint(spectrum, fcutmax=0.005, maxscan=1)
        register_result(regtest, r)
        res.append(r)
    output_test_result("exptail_noscan_multi", res)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_exptail_scan(regtest, name):
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    res = estimate_acfint(spectrum, fcutmax=0.03, maxscan=10)
    register_result(regtest, res)
    output_test_result(f"exptail_scan_{name}", res)


@pytest.mark.parametrize("names", NAME_LISTS)
@pytest.mark.parametrize("model", [ExpTailModel(), WhiteNoiseModel()])
@pytest.mark.parametrize("zero_freq", [False, True])
def test_scan_multi(regtest, zero_freq, model, names):
    should_warn = names[0].startswith("double") and isinstance(model, WhiteNoiseModel)
    res = []
    for name in names:
        spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
        if not zero_freq:
            spectrum = spectrum.without_zero_freq()
        with ExitStack() as stack:
            if should_warn:
                stack.enter_context(pytest.warns(FCutWarning))
            r = estimate_acfint(spectrum, fcutmax=0.01, maxscan=10, model=model)
        register_result(regtest, r)
        res.append(r)
    prefix = f"{model.name}_scan_multi"
    if not zero_freq:
        prefix += "_nodc"
    output_test_result(prefix, res)
