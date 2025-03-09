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

from stacie.estimate import Result, estimate_acint
from stacie.model import ChebyshevModel, ExpTailModel, SpectrumModel
from stacie.msgpack import dump, load
from stacie.plot import plot_results
from stacie.spectrum import Spectrum
from stacie.summary import summarize_results

DOUBLE_NAMES = ["double1", "double2"]
WHITE_NAMES = ["white1", "white2"]
NAME_LISTS = DOUBLE_NAMES, WHITE_NAMES
ALL_NAMES = [j for i in NAME_LISTS for j in i]


def output_test_result(prefix: str, res: Result | list[Result]):
    dn_out = Path("tests/outputs")
    dn_out.makedirs_p()
    plot_results(dn_out / f"{prefix}.pdf", res)
    with open(dn_out / f"{prefix}.txt", "w") as fh:
        fh.write(summarize_results(res))
    path_zip = dn_out / f"{prefix}.nmpk.xz"
    dump(path_zip, res)


def register_result(regtest, res: Result, white: bool = False):
    """Register the result of a test with regtest.

    Parameters
    ----------
    res
        The ``stacie.estimate.Result`` to register.
    white
        Whether the test uses a white-noise model or white-noise data.
        In this case, there is no point in writing the exponential correlation time.
    """
    with regtest:
        print(f"acint = {res.acint:.5e} ± {res.acint_std:.5e}")
        if not white and "corrtime_exp" in res.props:
            # Do not check flaky results, as they will depend on irrelevant details,
            # like CPU architecture and NumPy version.
            print(
                f"corrtime exp = {res.props['corrtime_exp']:.5e} "
                f"± {res.props['corrtime_exp_std']:.5e}"
            )
        print(f"corrtime int = {res.corrtime_int:.5e} ± {res.corrtime_int_std:.5e}")
        print(f"log(lh) = {res.props['ll']:.5e}")
        print("---")


def check_noscan_single(
    regtest,
    spectrum: Spectrum,
    prefix: str,
    *,
    fcutmax: float = 0.005,
    model: SpectrumModel | None = None,
):
    res = estimate_acint(spectrum, fcutmax=fcutmax, maxscan=1, model=model)
    register_result(regtest, res, "white" in prefix or isinstance(model, ChebyshevModel))
    output_test_result(prefix, res)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_exptail_noscan(regtest, name: str):
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    check_noscan_single(regtest, spectrum, f"exptail_noscan_{name}")


@pytest.mark.parametrize("name", ALL_NAMES)
def test_exptail_noscan_conditioning(regtest, name: str):
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    scale = 1e-15
    spectrum.amplitudes *= scale
    spectrum.variance *= scale
    check_noscan_single(regtest, spectrum, f"exptail_noscan_{name}_conditioning")


@pytest.mark.parametrize("name", WHITE_NAMES)
def test_white_noscan(regtest, name: str):
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    check_noscan_single(
        regtest, spectrum, f"white_noscan_{name}", fcutmax=0.1, model=ChebyshevModel(0)
    )


@pytest.mark.parametrize(("name", "fcutmax"), [("white2", 0.008), ("double1", 0.05)])
def test_exptail_noscan_fail(regtest, name: str, fcutmax: float):
    # Try cutoffs for which the convergence of the ExpTail parameters is known to be problematic.
    # Stacie should still produce some output without raising exceptions.
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    check_noscan_single(regtest, spectrum, f"exptail_noscan_{name}_fail", fcutmax=fcutmax)


@pytest.mark.parametrize("names", NAME_LISTS)
def test_exptail_noscan_multi(regtest, names: list[str]):
    res = []
    for name in names:
        spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
        r = estimate_acint(spectrum, fcutmax=0.005, maxscan=1)
        register_result(regtest, r, "white" in name)
        res.append(r)
    output_test_result("exptail_noscan_multi", res)


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("fcutmax", [0.03, None])
def test_exptail_scan(regtest, fcutmax: float, name: list[str]):
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    if fcutmax is None:
        # By dropping the DC component,
        # the number of frequencies becomes equal to nfitmax_hard.
        # This means spectrum.freqs[nfit] will not work.
        # Any occurrence of this will raise an error.
        spectrum = spectrum.without_zero_freq()
    res = estimate_acint(spectrum, fcutmax=fcutmax, maxscan=10)
    if not (fcutmax is None and "white" in name):
        # Only perform the regresion test for well-behaved cases.
        register_result(regtest, res, "white" in name)
    prefix = f"exptail_scan_{name}"
    if fcutmax is None:
        prefix += "_nofcut"
    output_test_result(prefix, res)


@pytest.mark.parametrize("names", NAME_LISTS)
@pytest.mark.parametrize("model", [ExpTailModel(), ChebyshevModel(0)])
@pytest.mark.parametrize("zero_freq", [False, True])
def test_scan_multi(regtest, zero_freq: bool, model: SpectrumModel, names: list[str]):
    res = []
    for name in names:
        spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
        if not zero_freq:
            spectrum = spectrum.without_zero_freq()
        r = estimate_acint(spectrum, fcutmax=0.01, maxscan=10, model=model)
        register_result(regtest, r, "white" in name or isinstance(model, ChebyshevModel))
        res.append(r)
    prefix = f"{model.name}_scan_multi"
    if not zero_freq:
        prefix += "_nodc"
    output_test_result(prefix, res)
