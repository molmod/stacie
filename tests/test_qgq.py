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

from pathlib import Path

import numdifftools as nd
import numpy as np
import pytest
import scipy as sp

from stacie.qgq import (
    construct_qgq_empirical,
    construct_qgq_stdnormal,
    kld_constr,
    plot_qgq,
    solve_kld,
    solve_quadratic,
)


@pytest.mark.parametrize("solver", [solve_quadratic, solve_kld])
def test_solve_gradient(solver):
    rhs = np.zeros(3)
    rhs[0] = 1
    wref = np.ones(8)
    wref[0] = 0.5

    def costfn(x):
        eqs = np.array([np.ones_like(x), x, x**2 - 1])
        eqs_d = np.array([np.zeros_like(x), np.ones_like(x), 2 * x])
        cost, _, grad = solver(eqs, eqs_d, rhs, wref)
        return cost, grad

    x = np.linspace(-3, 3, 8)
    grad = costfn(x)[1]
    grad_num = nd.Gradient(lambda x: costfn(x)[0])(x)
    assert grad == pytest.approx(grad_num, abs=1e-5)


def test_kld_constr_jac():
    rng = np.random.default_rng(0)
    eqs = rng.normal(size=(3, 8))
    rhs = rng.normal(size=3)
    wref = rng.uniform(1, 2, size=8)

    def constrfn(lbda):
        errors, jac = kld_constr(lbda, eqs, rhs, wref)
        return errors, jac

    lbda = rng.normal(size=3)
    _, jac = constrfn(lbda)
    jac_num = nd.Jacobian(lambda lbda: kld_constr(lbda, eqs, rhs, wref)[0])(lbda)
    assert jac == pytest.approx(jac_num, abs=1e-5)


@pytest.mark.parametrize("zero", [True, False])
@pytest.mark.parametrize("solver", [solve_quadratic, solve_kld])
def test_construct_qgq_stdnormal(zero, solver):
    points0 = np.linspace(0, 1.5, 5)
    if not zero:
        points0[0] = 0.2
    points, weights, extra = construct_qgq_stdnormal(
        points0, nmoment=2, do_extra=True, weight_solver=solver
    )
    assert len(points) == len(weights) == 2 * len(points0) - zero
    assert np.all(weights > 0)
    assert weights.sum() == pytest.approx(1, abs=1e-5)
    mu1 = np.dot(weights, points)
    assert mu1 == pytest.approx(0, abs=1e-5)
    mu2 = np.dot(weights, points**2)
    assert mu2 == pytest.approx(1, abs=1e-5)

    # Write plot
    def plot_dist(ax):
        ax.set_title(f"Standard normal distribution {solver.__name__}, zero={zero}")
        xgrid = np.linspace(-3, 3, 100)
        ax.plot(xgrid, sp.stats.norm.pdf(xgrid), "k-")
        ax.plot(points, sp.stats.norm.pdf(points), "ro")
        ax.set_xlabel("x")
        ax.set_ylabel("density")

    dn_out = Path("tests/outputs")
    dn_out.mkdir(exist_ok=True)
    fig, _ = plot_qgq(extra, plot_dist)
    stem = f"qgq_stdnormal_{solver.__name__}"
    if zero:
        stem += "_zero"
    fig.savefig(dn_out / f"{stem}.pdf")


@pytest.mark.parametrize("nmoment", [4, 5])
@pytest.mark.parametrize("guess", [True, False])
def test_construct_qgq_empirical(nmoment, guess):
    rng = np.random.default_rng(0)
    shape = 20
    scale = 1.2
    samples = rng.gamma(shape=shape, scale=scale, size=1000)
    points0 = 20 if guess else np.quantile(samples, (np.arange(-10, 10) + 0.5) / 20 + 0.5)
    points, weights, extra = construct_qgq_empirical(
        samples, points0=points0, nmoment=nmoment, do_extra=True
    )
    assert len(points) == len(weights) == 20
    assert np.all(weights > 0)
    assert weights.sum() == pytest.approx(1, abs=1e-5)
    mu1 = np.dot(weights, points)
    assert mu1 == pytest.approx(samples.mean(), abs=1e-5)
    mu2 = np.dot(weights, (points - mu1) ** 2)
    assert mu2 == pytest.approx(samples.var(), abs=1e-5)
    mu3 = np.dot(weights, (points - mu1) ** 3)
    skew = mu3 / mu2**1.5
    assert skew == pytest.approx(sp.stats.skew(samples), abs=1e-5)
    mu4 = np.dot(weights, (points - mu1) ** 4)
    kurt = mu4 / mu2**2
    assert kurt == pytest.approx(sp.stats.kurtosis(samples, fisher=False), abs=1e-5)

    # write plot
    def plot_dist(ax):
        ax.set_title(f"Gamma $α={shape}$, $θ={scale}$, $S=1000$, $M={nmoment}$")
        ax.hist(samples, bins=30, density=True, alpha=0.7)
        xgrid = np.linspace(samples.min(), samples.max(), 100)
        mygamma = sp.stats.gamma(a=shape, scale=scale)
        ax.plot(xgrid, mygamma.pdf(xgrid), "k-")
        ax.plot(points, mygamma.pdf(points), "ro")
        ax.set_xlabel("x")
        ax.set_ylabel("density")

    dn_out = Path("tests/outputs")
    dn_out.mkdir(exist_ok=True)
    fig, _ = plot_qgq(extra, plot_dist)
    stem = f"qgq_empirical_{nmoment}"
    if guess:
        stem += "_guess"
    fig.savefig(dn_out / f"{stem}.pdf")
