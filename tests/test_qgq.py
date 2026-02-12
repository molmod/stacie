# STACIE is a STable AutoCorrelation Integral Estimator.
# Copyright 2024-2026 The contributors of the STACIE Python Package.
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
"""Tests for ``stacie.qgq``"""

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import pytest
import scipy as sp

from stacie.qgq import (
    assign_weights_l2,
    construct_qgq_empirical,
    construct_qgq_stdnormal,
    plot_qgq,
    point_cost,
)


def test_assign_weights_l2():
    x = np.linspace(-1, 1, 8)
    eqs = np.array([x, x**2 - 1, x**3 - 3 * x])
    rhs = np.zeros(3)
    wref = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    weights = assign_weights_l2(eqs, rhs, wref)
    assert weights.sum() == pytest.approx(1, abs=1e-5)
    assert np.dot(eqs, weights) == pytest.approx(rhs, abs=1e-5)


@pytest.mark.parametrize("eps", [0.0, 0.1, 1.0])
def test_point_cost_jac(eps):
    rng = np.random.default_rng(0)
    points = rng.normal(size=8)
    basis_funcs = [lambda x: np.ones_like(x), lambda x: x, lambda x: x**2 - 1]
    basis_funcs_d = [lambda x: np.zeros_like(x), lambda x: np.ones_like(x), lambda x: 2 * x]
    target_basis = rng.uniform(0, 1, size=3)
    targets_weights = rng.uniform(1, 2, size=8)
    mycost = partial(
        point_cost,
        basis_funcs=basis_funcs,
        basis_funcs_d=basis_funcs_d,
        targets_basis=target_basis,
        targets_weights=targets_weights,
        eps=eps,
    )

    grad = mycost(points)[1]
    grad_num = nd.Gradient(lambda points: mycost(points)[0])(points)
    assert grad == pytest.approx(grad_num, abs=1e-5)


HARD_CASES_STDNORMAL = [(10, 6), (10, 8), (20, 8), (30, 8)]


@pytest.mark.parametrize("zero", [True, False])
@pytest.mark.parametrize("npoint", [10, 20, 30, 50, 100])
@pytest.mark.parametrize("nmoment", [2, 4, 6, 8])
def test_construct_qgq_stdnormal(zero, npoint, nmoment):
    points0 = np.linspace(0, 1.5, npoint)
    if not zero:
        points0 += 0.2
    # gtol is relaxed to speed up the test. Avoid this in production.
    points, weights, extra = construct_qgq_stdnormal(
        points0, nmoment=nmoment, gtol=1e-4, do_extra=True
    )
    assert len(points) == len(weights) == 2 * len(points0) - zero
    assert np.diff(points).min() > 0
    if (npoint, nmoment) not in HARD_CASES_STDNORMAL:
        assert np.all(weights > 0)
    assert weights.sum() == pytest.approx(1, abs=1e-5)
    mu1 = np.dot(weights, points)
    assert mu1 == pytest.approx(0, abs=1e-5)
    mu2 = np.dot(weights, points**2)
    assert mu2 == pytest.approx(1, abs=1e-5)

    # Write plot
    def plot_dist(ax):
        ax.set_title(f"Std normal, zero={zero}, N={npoint}, M={nmoment}")
        xgrid = np.linspace(-3, 3, 100)
        ax.plot(xgrid, sp.stats.norm.pdf(xgrid), "k-")
        ax.plot(points, sp.stats.norm.pdf(points), "r+")
        ax.set_xlabel("x")
        ax.set_ylabel("density")

    dn_out = Path("tests/outputs")
    dn_out.mkdir(exist_ok=True)
    fig, _ = plot_qgq(extra, plot_dist)
    stem = f"qgq_stdnormal_{npoint}_{nmoment}"
    if zero:
        stem += "_zero"
    fig.savefig(dn_out / f"{stem}.pdf")
    plt.close(fig)


HARD_CASES_EMPIRICAL = [(10, 6), (20, 6)]


@pytest.mark.parametrize("guess", [True, False])
@pytest.mark.parametrize("npoint", [10, 20, 30, 50, 100])
@pytest.mark.parametrize("nmoment", [2, 3, 4, 5, 6])
def test_construct_qgq_empirical(guess, npoint, nmoment):
    rng = np.random.default_rng(0)
    shape = 20
    scale = 1.2
    samples = rng.gamma(shape=shape, scale=scale, size=1000)
    if guess:
        points0 = npoint
    else:
        points0 = np.quantile(
            samples, (np.arange(-npoint // 2, npoint // 2) + 0.5) / (npoint + 2) + 0.5
        )
    # gtol is relaxed to speed up the test. Avoid this in production.
    points, weights, extra = construct_qgq_empirical(
        samples, points0=points0, nmoment=nmoment, gtol=1e-4, do_extra=True
    )
    assert len(points) == len(weights) == npoint
    assert np.diff(points).min() > 0
    if (npoint, nmoment) not in HARD_CASES_EMPIRICAL:
        assert np.all(weights > 0)
    assert weights.sum() == pytest.approx(1, abs=1e-5)
    mu1 = np.dot(weights, points)
    assert mu1 == pytest.approx(samples.mean(), abs=1e-5)
    mu2 = np.dot(weights, (points - mu1) ** 2)
    assert mu2 == pytest.approx(samples.var(), abs=1e-5)
    if nmoment >= 3:
        mu3 = np.dot(weights, (points - mu1) ** 3)
        skew = mu3 / mu2**1.5
        assert skew == pytest.approx(sp.stats.skew(samples), abs=1e-5)
    if nmoment >= 4:
        mu4 = np.dot(weights, (points - mu1) ** 4)
        kurt = mu4 / mu2**2
        assert kurt == pytest.approx(sp.stats.kurtosis(samples, fisher=False), abs=1e-5)

    # write plot
    def plot_dist(ax):
        ax.set_title(f"Gamma α={shape}, θ={scale}, S=1000, N={npoint}, M={nmoment}")
        ax.hist(samples, bins=30, density=True, alpha=0.7)
        xgrid = np.linspace(samples.min(), samples.max(), 100)
        mygamma = sp.stats.gamma(a=shape, scale=scale)
        ax.plot(xgrid, mygamma.pdf(xgrid), "k-")
        ax.plot(points, mygamma.pdf(points), "r+")
        ax.set_xlabel("x")
        ax.set_ylabel("density")

    dn_out = Path("tests/outputs")
    dn_out.mkdir(exist_ok=True)
    fig, _ = plot_qgq(extra, plot_dist)
    stem = f"qgq_empirical_{npoint}_{nmoment}"
    if guess:
        stem += "_guess"
    fig.savefig(dn_out / f"{stem}.pdf")
    plt.close(fig)
