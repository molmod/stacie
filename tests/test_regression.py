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
"""Regression tests for typical Stacie workflows.

Not that some regressions are not tracked
because the algorithm is known and expected to be flaky for specific combinations of inputs.
"""

import pytest
from path import Path
from stacie.estimate import estimate_acfint
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
        print(f"ACF Int  = {res.props['acfint']:.5e} ± {res.props['acfint_std']:.5e}")
        print(f"Tau tail = {res.props['corrtime_tail']:.5e} ± {res.props['corrtime_tail_std']:.5e}")
        print(f"Log lh   = {res.props['ll']:.5e}")
        print("---")


def check_noscan_single(
    regtest,
    spectrum: Spectrum,
    prefix: str,
    *,
    fcutmax: float = 0.005,
    model: SpectrumModel | None = None,
    flaky: bool = False,
):
    res = estimate_acfint(spectrum, fcutmax=fcutmax, maxscan=1, model=model)
    if not flaky:
        register_result(regtest, res)
    output_test_result(prefix, res)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_exptail_noscan(regtest, name):
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    flaky = name == "white1"
    check_noscan_single(regtest, spectrum, f"exptail_noscan_{name}", flaky=flaky)


@pytest.mark.parametrize("name", WHITE_NAMES)
def test_white_noscan(regtest, name):
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    check_noscan_single(
        regtest, spectrum, f"white_noscan_{name}", fcutmax=0.1, model=WhiteNoiseModel()
    )


@pytest.mark.parametrize(("name", "fcutmax"), [("white2", 0.008), ("double1", 0.05)])
def test_exptail_noscan_fail(regtest, name, fcutmax):
    # Try cutoffs for which the convergence of the ExpTail parameters is known to be problematic.
    # Stacie should still produce some output without raising exceptions.
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    check_noscan_single(
        regtest, spectrum, f"exptail_noscan_{name}_fail", fcutmax=fcutmax, flaky=True
    )


@pytest.mark.parametrize("names", NAME_LISTS)
def test_exptail_noscan_multi(regtest, names):
    flaky = names[0].startswith("white")
    res = []
    for name in names:
        spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
        r = estimate_acfint(spectrum, fcutmax=0.005, maxscan=1)
        if not flaky:
            register_result(regtest, r)
        res.append(r)
    output_test_result("exptail_noscan_multi", res)


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("fcutmax", [0.03, None])
def test_exptail_scan(regtest, fcutmax, name):
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    if fcutmax is None:
        # By dropping the DC component,
        # the number of frequencies becomes equal to ncutmax_hard.
        # This means spectrum.freqs[ncut] will not work.
        # Any occurrence of this will raise an error.
        spectrum = spectrum.without_zero_freq()
    flaky = fcutmax is None and name == "white1"
    res = estimate_acfint(spectrum, fcutmax=fcutmax, maxscan=10)
    if not flaky:
        register_result(regtest, res)
    prefix = f"exptail_scan_{name}"
    if fcutmax is None:
        prefix += "_nofcut"
    output_test_result(prefix, res)


@pytest.mark.parametrize("names", NAME_LISTS)
@pytest.mark.parametrize("model", [ExpTailModel(), WhiteNoiseModel()])
@pytest.mark.parametrize("zero_freq", [False, True])
def test_scan_multi(regtest, zero_freq, model, names):
    flaky = not zero_freq and isinstance(model, ExpTailModel) and names[0].startswith("white")
    res = []
    for name in names:
        spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
        if not zero_freq:
            spectrum = spectrum.without_zero_freq()
        r = estimate_acfint(spectrum, fcutmax=0.01, maxscan=10, model=model)
        if not flaky:
            register_result(regtest, r)
        res.append(r)
    prefix = f"{model.name}_scan_multi"
    if not zero_freq:
        prefix += "_nodc"
    output_test_result(prefix, res)
