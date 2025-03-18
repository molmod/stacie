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
"""Regression tests for typical Stacie workflows.

Note that the test cases here are not meant as examples of properly analyzed spectra.
(Some of the fits are really poor.)
These test cases explore a few corner cases and verify that Stacie behaves as expected.
"""

from collections.abc import Callable

import pytest
from path import Path

from stacie.cutoff import halfhalf_criterion, underfitting_criterion
from stacie.estimate import Result, estimate_acint
from stacie.model import ChebyshevModel, ExpTailModel, SpectrumModel
from stacie.msgpack import dump, load
from stacie.plot import plot_results
from stacie.spectrum import Spectrum
from stacie.summary import summarize_results

# Combinations of test spectra and suitable models, with a manual cutoff frequency.
CASES = [
    ("white", ChebyshevModel(0), 0.1),
    ("white", ChebyshevModel(2), 0.1),
    ("broad", ChebyshevModel(0), 0.01),
    ("broad", ChebyshevModel(2), 0.1),
    ("pure", ExpTailModel(), 0.02),
    ("pure", ChebyshevModel(2, even=True), 0.02),
    ("double", ExpTailModel(), 0.02),
    ("double", ChebyshevModel(2, even=True), 0.02),
]
CRITERIA = [halfhalf_criterion, underfitting_criterion]


def output_test_result(prefix: str, res: Result | list[Result]):
    """Dump results with Stacie's standard output formats."""
    dn_out = Path("tests/outputs")
    dn_out.makedirs_p()
    plot_results(dn_out / f"{prefix}.pdf", res)
    with open(dn_out / f"{prefix}.txt", "w") as fh:
        fh.write(summarize_results(res))
    path_zip = dn_out / f"{prefix}.nmpk.xz"
    dump(path_zip, res)


def register_result(regtest, res: Result):
    """Register the result of a test with regtest.

    Parameters
    ----------
    res
        The ``stacie.estimate.Result`` to register.
    """
    with regtest:
        print(f"acint = {res.acint:.5e} ± {res.acint_std:.5e}")
        if "corrtime_exp" in res.props:
            print(
                f"corrtime exp = {res.props['corrtime_exp']:.5e} "
                f"± {res.props['corrtime_exp_std']:.5e}"
            )
        print(f"corrtime int = {res.corrtime_int:.5e} ± {res.corrtime_int_std:.5e}")
        print(f"log(lh) = {res.props['ll']:.5e}")
        print("---")


@pytest.mark.parametrize(("name", "model", "fcutmax"), CASES)
@pytest.mark.parametrize("criterion", CRITERIA)
@pytest.mark.parametrize("full", [True, False])
@pytest.mark.filterwarnings("ignore::stacie.estimate.FCutWarning")
def test_case_scan(
    regtest, name: str, model: SpectrumModel, fcutmax: float, criterion: Callable, full: bool
):
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    if name == "broad":
        spectrum = spectrum.without_zero_freq()
    result = estimate_acint(
        spectrum,
        model,
        fcutmax=None if full else fcutmax * 2,
        nfitmax_hard=10000,
        maxscan=10,
        cutoff_criterion=criterion,
    )
    register_result(regtest, result)
    prefix = f"scan_{name}_{model.name}_{criterion.__name__}"
    output_test_result(prefix, result)


@pytest.mark.parametrize(("name", "model", "fcutmax"), CASES)
def test_case_noscan(regtest, name: str, model: SpectrumModel, fcutmax: float):
    spectrum = load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum)
    result = estimate_acint(
        spectrum,
        model,
        fcutmax=fcutmax,
        maxscan=1,
    )
    register_result(regtest, result)
    prefix = f"noscan_{name}_{model.name}"
    output_test_result(prefix, result)


@pytest.mark.parametrize("scan", [True, False])
def test_plot_multi(scan: bool):
    cases = [
        ("white", ChebyshevModel(2), 0.1),
        ("broad", ChebyshevModel(2), 0.1),
    ]
    results = [
        estimate_acint(
            load(f"tests/inputs/spectrum_{name}.nmpk.xz", Spectrum),
            model,
            fcutmax=2 * fcutmax if scan else fcutmax,
            maxscan=10 if scan else 1,
            nfitmax_hard=10000,
        )
        for name, model, fcutmax in cases
    ]
    prefix = "multin_" + ("scan" if scan else "noscan")
    output_test_result(prefix, results)
