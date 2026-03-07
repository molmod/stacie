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
"""Tests for ``stacie.dq``"""

from pathlib import Path

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import pytest
import scipy as sp

from stacie.dq import (
    Equations,
    Symmetry,
    construct_dq_empirical,
    construct_dq_stdnormal,
    dq3,
    plot_dq,
)


def make_simple_eqs(symmetry):
    funcs = [lambda x: x, lambda x: x**2 - 1]
    funcs_d = [np.ones_like, lambda x: 2 * x]
    funcs_dd = [np.zeros_like, lambda x: np.full_like(x, 2)]
    targets = np.array([0.0, 0.0])
    points0 = np.array([2.0, 2.5, 3.0])
    if symmetry == Symmetry.ZERO:
        weights0 = np.array([0.1, 0.3, 0.4, 0.5])
    else:
        weights0 = np.array([0.1, 0.2, 0.3])
    return points0, Equations(
        funcs=funcs,
        funcs_d=funcs_d,
        funcs_dd=funcs_dd,
        targets=targets,
        weights0=weights0,
        symmetry=symmetry,
    )


@pytest.mark.parametrize("symmetry", [Symmetry.NONE, Symmetry.ZERO, Symmetry.NONZERO])
def test_apply_sym(symmetry):
    points, eqs = make_simple_eqs(symmetry)
    weights = eqs.weights0
    points_sym, weights_sym = eqs.apply_sym(points)
    if symmetry == Symmetry.NONE:
        assert points_sym == pytest.approx(points)
        assert weights_sym == pytest.approx(weights)
    elif symmetry == Symmetry.ZERO:
        assert len(points_sym) == 2 * len(points) + 1
        assert len(weights_sym) == 2 * len(points) + 1
        assert -points_sym[: len(points)][::-1] == pytest.approx(points)
        assert points_sym[len(points)] == pytest.approx(0.0)
        assert points_sym[len(points) + 1 :] == pytest.approx(points)
        assert weights_sym[: len(points)][::-1] == pytest.approx(weights[1:])
        assert weights_sym[len(points)] == pytest.approx(2 * weights[0])
        assert weights_sym[len(points) + 1 :] == pytest.approx(weights[1:])
    elif symmetry == Symmetry.NONZERO:
        assert len(points_sym) == 2 * len(points)
        assert len(weights_sym) == 2 * len(points)
        assert -points_sym[: len(points)][::-1] == pytest.approx(points)
        assert points_sym[len(points) :] == pytest.approx(points)
        assert weights_sym[: len(points)][::-1] == pytest.approx(weights)
        assert weights_sym[len(points) :] == pytest.approx(weights)


@pytest.mark.parametrize("symmetry", [Symmetry.NONE, Symmetry.ZERO, Symmetry.NONZERO])
def test_backprop_sym(symmetry):
    points0, eqs = make_simple_eqs(symmetry)
    points_sym, _ = eqs.apply_sym(points0)
    jac2 = np.linspace(0, 1, 10 * len(points_sym)).reshape(-1, len(points_sym))
    jac = eqs.backprop_sym(jac2)
    assert jac.shape == (10, len(points0))
    if symmetry == Symmetry.NONE:
        assert jac == pytest.approx(jac2)
    else:
        assert jac == pytest.approx(-jac2[:, : len(points0)][:, ::-1] + jac2[:, -len(points0) :])


@pytest.mark.parametrize("symmetry", [Symmetry.NONE, Symmetry.ZERO, Symmetry.NONZERO])
def test_jacobian_eqs_sym(symmetry):
    points0, eqs = make_simple_eqs(symmetry)
    points0_sym, _ = eqs.apply_sym(points0)
    eqs.weights0[:] = 1.0
    eqs0, jac_an = eqs(points0, deriv=1)
    assert len(eqs0) == len(eqs.funcs) + len(points0_sym)
    assert jac_an.shape == (len(eqs.funcs) + len(points0_sym), len(points0))
    jac_num = nd.Jacobian(lambda x: eqs(x, deriv=0)[0])(points0)
    assert jac_an == pytest.approx(jac_num, abs=1e-5)


def test_jacobian_eqs():
    points0, eqs = make_simple_eqs(Symmetry.NONE)
    eqs0, jac_an = eqs.compute_low(points0, eqs.weights0, deriv=1)
    assert len(eqs0) == len(eqs.funcs) + len(points0)
    assert jac_an.shape == (len(eqs.funcs) + len(points0), len(points0))
    jac_num = nd.Jacobian(lambda x: eqs.compute_low(x, eqs.weights0, deriv=0)[0])(points0)
    assert jac_an == pytest.approx(jac_num, abs=1e-5)


def test_proj():
    points0, eqs = make_simple_eqs(Symmetry.NONE)
    mat1, apply_proj = eqs.proj(points0, eqs.weights0, deriv=0)
    assert mat1.shape == (len(eqs.funcs), len(points0))
    _u, _s, vt = np.linalg.svd(mat1, full_matrices=False)
    proj = np.eye(len(points0)) - np.dot(vt.T, vt)

    # Test apply_proj with vector
    rng = np.random.default_rng(0)
    f = rng.normal(size=len(points0))
    g = apply_proj(f)
    assert g.shape == f.shape
    assert np.allclose(g, np.dot(proj, f))

    # Test apply_proj with matrix
    f = rng.normal(size=(len(points0), 5))
    g = apply_proj(f)
    assert g.shape == f.shape
    assert np.allclose(g, np.dot(proj, f))


def test_jacobian_proj():
    points0, eqs = make_simple_eqs(Symmetry.NONE)

    rng = np.random.default_rng(0)
    f = rng.normal(size=len(points0))

    def func_proj(x):
        _mat1, apply_proj = eqs.proj(x, eqs.weights0, deriv=0)
        return apply_proj(f)

    def func_proj_d(x):
        apply_proj_d = eqs.proj(x, eqs.weights0, deriv=1)[2]
        return apply_proj_d(f)

    p_d_an = func_proj_d(points0)
    p_d_num = nd.Jacobian(func_proj)(points0)
    assert p_d_an == pytest.approx(p_d_num, abs=1e-5)


def test_gradient_penalty():
    weights = np.linspace(1, 2, 8)
    points = np.linspace(-1, 1, 8)
    eqs = Equations(
        funcs=[],
        funcs_d=[],
        funcs_dd=[],
        targets=np.array([]),
        weights0=np.array([]),
        symmetry=Symmetry.NONE,
    )
    grad_an = eqs.pen(points, weights, deriv=1)[1]
    grad_num = nd.Gradient(lambda x: eqs.pen(x, weights, deriv=0)[0])(points)
    assert grad_an == pytest.approx(grad_num, abs=1e-5)


def test_hessian_penalty():
    weights = np.linspace(1, 2, 8)
    points = np.linspace(-1, 1, 8)
    eqs = Equations(
        funcs=[],
        funcs_d=[],
        funcs_dd=[],
        targets=np.array([]),
        weights0=np.array([]),
        symmetry=Symmetry.NONE,
    )
    hess_an = eqs.pen(points, weights, deriv=2)[2]
    hess_num = nd.Jacobian(lambda x: eqs.pen(x, weights, deriv=1)[1])(points)
    assert hess_an == pytest.approx(hess_num, abs=1e-5)


def test_solve_weights():
    points = np.linspace(-1, 1, 8)
    funcs = [lambda x: x, lambda x: x**2 - 1]
    targets = np.array([0.0, 0.0])
    eqs = Equations(
        funcs=funcs,
        funcs_d=[],
        funcs_dd=[],
        targets=targets,
        weights0=np.ones_like(points),
        symmetry=Symmetry.NONE,
    )
    weights = eqs.solve_weights(points, eqs.weights0)
    assert weights.sum() == pytest.approx(1, abs=1e-5)
    for f, t in zip(funcs, targets, strict=True):
        assert np.dot(f(points), weights) == pytest.approx(t, abs=1e-5)


@pytest.mark.parametrize("symmetry", [Symmetry.ZERO, Symmetry.NONZERO])
@pytest.mark.parametrize(
    ("npoint", "nmoment"),
    [
        (10, 2),
        (10, 4),
        # (10, 6),  # unsolvable
        # (10, 8),  # unsolvable
        (20, 2),
        (20, 4),
        (20, 6),
        # (20, 8),  # unsolvable
        (30, 2),
        (30, 4),
        (30, 6),
        # (30, 8),  # unsolvable
        (50, 2),
        (50, 4),
        (50, 6),
        (50, 8),
        (100, 2),
        (100, 4),
        (100, 6),
        (100, 8),
    ],
)
def test_construct_dq_stdnormal(symmetry, npoint, nmoment):
    points0 = np.linspace(0.1, 1.5, npoint)
    points, weights, extra = construct_dq_stdnormal(
        points0, nmoment=nmoment, symmetry=symmetry, verbose=True, do_extra=True
    )
    assert len(points) == len(weights) == 2 * len(points0) + int(symmetry == Symmetry.ZERO)
    assert np.diff(points).min() > 0
    assert weights.sum() == pytest.approx(1, abs=1e-5)
    mu1 = np.dot(weights, points)
    assert mu1 == pytest.approx(0, abs=1e-5)
    mu2 = np.dot(weights, points**2)
    assert mu2 == pytest.approx(1, abs=1e-5)

    # Write plot
    def plot_dist(ax):
        ax.set_title(f"Std normal, zero={symmetry == Symmetry.ZERO}, N={npoint}, M={nmoment}")
        xgrid = np.linspace(-3, 3, 100)
        ax.plot(xgrid, sp.stats.norm.pdf(xgrid), "k-")
        ax.plot(points, sp.stats.norm.pdf(points), "r+")
        ax.set_xlabel("x")
        ax.set_ylabel("density")

    dn_out = Path("tests/outputs")
    dn_out.mkdir(exist_ok=True)
    fig, _ = plot_dq(extra, plot_dist)
    stem = f"dq_stdnormal_{npoint}_{nmoment}"
    if symmetry == Symmetry.ZERO:
        stem += "_zero"
    fig.savefig(dn_out / f"{stem}.pdf")
    plt.close(fig)


@pytest.mark.parametrize("guess", [True, False])
@pytest.mark.parametrize(
    ("npoint", "nmoment"),
    [
        (10, 2),
        (10, 3),
        (10, 4),
        (10, 5),
        # (10, 6),  # unsolvable
        (20, 2),
        (20, 3),
        (20, 4),
        (20, 5),
        # (20, 6),  # unsolvable
        (30, 2),
        (30, 3),
        (30, 4),
        (30, 5),
        (30, 6),
        (50, 2),
        (50, 3),
        (50, 4),
        (50, 5),
        (50, 6),
        (100, 2),
        (100, 3),
        (100, 4),
        (100, 5),
        (100, 6),
    ],
)
def test_construct_dq_empirical(guess, npoint, nmoment):
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
    points, weights, extra = construct_dq_empirical(
        samples, points0=points0, nmoment=nmoment, verbose=True, do_extra=True
    )
    assert len(points) == len(weights) == npoint
    assert np.diff(points).min() > 0
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
    fig, _ = plot_dq(extra, plot_dist)
    stem = f"dq_empirical_{npoint}_{nmoment}"
    if guess:
        stem += "_guess"
    fig.savefig(dn_out / f"{stem}.pdf")
    plt.close(fig)


def test_dq_stdnormal_design():
    x_design, w_design = construct_dq_stdnormal(
        np.linspace(0.1, 1.0, 3), 6, Symmetry.NONZERO, weights0=[12, 2, 1], verbose=True
    )
    print(w_design)
    assert len(x_design) == len(w_design) == 6
    assert np.diff(x_design).min() > 0
    assert w_design.min() > 0
    assert w_design / np.array([1, 2, 12, 12, 2, 1]) == pytest.approx(1 / 30, abs=1e-5)
    assert w_design.sum() == pytest.approx(1, abs=1e-5)
    mu1 = np.dot(w_design, x_design)
    assert mu1 == pytest.approx(0, abs=1e-5)
    mu2 = np.dot(w_design, x_design**2)
    assert mu2 == pytest.approx(1, abs=1e-5)


def test_dq3_sym():
    points = dq3(0, 1, 0)
    assert (points == points.real).all()
    assert points.mean() == pytest.approx(0, abs=1e-5)
    assert (points**2).mean() == pytest.approx(1, abs=1e-5)
    assert (points**3).mean() == pytest.approx(0, abs=1e-5)


def test_dq3_skew():
    points = dq3(0.5, 2.0, 0.7)
    assert (points == points.real).all()
    mu = points.mean()
    assert mu == pytest.approx(0.5, abs=1e-5)
    sigma = np.sqrt(((points - mu) ** 2).mean())
    assert sigma == pytest.approx(2.0, abs=1e-5)
    skew = ((points - mu) ** 3).mean() / sigma**3
    assert skew == pytest.approx(0.7, abs=1e-5)


def test_dq3_quad():
    shape = 10.0
    scale = 3.0
    gamma = sp.stats.gamma(a=shape, scale=scale)
    skew = gamma.stats(moments="s")
    assert abs(skew) < 1 / np.sqrt(2)
    points = dq3(gamma.mean(), gamma.std(), skew)
    assert (points == points.real).all()
    poly = np.polynomial.Polynomial([5, 4, 3, 2])
    # Manual computation of the integral of polynomial times Gamma density.
    quad1 = (
        poly.coef[0]
        + poly.coef[1] * scale * shape
        + poly.coef[2] * scale**2 * shape * (shape + 1)
        + poly.coef[3] * scale**3 * shape * (shape + 1) * (shape + 2)
    )
    # The three-point quadrature should give the same result.
    quad2 = np.mean(poly(points))
    assert quad2 == pytest.approx(quad1, abs=1e-5)
