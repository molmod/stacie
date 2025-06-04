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
"""Regression tests for typical STACIE workflows.

Note that the test cases here are not meant as examples of properly analyzed spectra.
(Some of the fits are really poor.)
These test cases explore a few corner cases and verify that STACIE behaves as expected.
"""

from collections.abc import Callable

import numpy as np
import pytest
from path import Path

from stacie.cutoff import CV2LCriterion
from stacie.estimate import Result, estimate_acint, summarize_results
from stacie.model import ExpPolyModel, PadeModel, SpectrumModel
from stacie.plot import plot_results
from stacie.spectrum import Spectrum

# Combinations of test spectra and suitable models, with a manual cutoff frequency.
CASES = [
    ("white", ExpPolyModel([0]), 0.1),
    ("white", ExpPolyModel([0, 1, 2]), 0.1),
    ("broad", ExpPolyModel([0]), 0.01),
    ("broad", ExpPolyModel([0, 1, 2]), 0.1),
    ("pure", PadeModel([0, 2], [2]), 0.1),
    ("pure", ExpPolyModel([0, 2]), 0.02),
    ("double", PadeModel([0, 2], [2]), 0.1),
    ("double", ExpPolyModel([0, 2]), 0.05),
    ("double", PadeModel([0, 2], [2]), 0.05),
]
CRITERIA = [CV2LCriterion()]


def output_test_result(prefix: str, res: Result | list[Result]):
    """Dump results with STACIE's standard output formats."""
    dn_out = Path("tests/outputs")
    dn_out.makedirs_p()
    plot_results(dn_out / f"{prefix}.pdf", res)
    with open(dn_out / f"{prefix}.txt", "w") as fh:
        fh.write(summarize_results(res))


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
            print(f"corrtime exp = {res.corrtime_exp:.5e} ± {res.corrtime_exp_std:.5e}")
        print(f"corrtime int = {res.corrtime_int:.5e} ± {res.corrtime_int_std:.5e}")
        print("---")


@pytest.mark.parametrize(("name", "model", "fcut_max"), CASES)
@pytest.mark.parametrize("criterion", CRITERIA)
@pytest.mark.parametrize("full", [True, False])
def test_case_scan(
    regtest, name: str, model: SpectrumModel, fcut_max: float, criterion: Callable, full: bool
):
    spectrum = Spectrum(**np.load(f"tests/inputs/spectrum_{name}.npz"))
    if name == "broad":
        spectrum = spectrum.without_zero_freq()
    result = estimate_acint(spectrum, model, fcut_max=2 * fcut_max, cutoff_criterion=criterion)
    register_result(regtest, result)
    prefix = f"scan_{name}_{model.name}_{criterion.name}"
    output_test_result(prefix, result)


def test_plot_multi():
    cases = [
        ("white", ExpPolyModel([0, 2]), 0.1),
        ("broad", ExpPolyModel([0, 2]), 0.1),
    ]
    results = [
        estimate_acint(
            Spectrum(**np.load(f"tests/inputs/spectrum_{name}.npz")),
            model,
            fcut_max=fcut_max,
            cutoff_criterion=CV2LCriterion(),
        )
        for name, model, fcut_max in cases
    ]
    output_test_result("multi", results)
