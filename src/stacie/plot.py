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
"""Plot the results of the estimate of the ACF integral."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from .estimate import Result

__all__ = ("plot",)


def plot(path_pdf: str, res: Result | list[Result], acf_unit: str = "A", time_unit: str = "s"):
    """Plot the results of the ACF integral estimation.

    Parameters
    ----------
    path_pdf
        The PDF file where all the figures are stored.
    res
        A single Results instance or a list of them.
        If the (first) result instance has model_spectrum set,
        theoretical expectations are included.
        When multiple results instances are given,
        only the first one is plotted in blue.
        All remaining ones are plotted in light grey.
    acf_unit
        A string with the unit of the autocorrelation function times the prefactor
    time_unit
        A string with the unit of the time axis, used for labels.
    """
    # Prepare res
    if isinstance(res, Result):
        res = [res]
    sfac = 2

    model_props = {"ls": "--", "color": "k", "alpha": 0.5}

    def plot_spectrum(ax, r: Result):
        nplot = 2 * r.ncut
        s = r.spectrum
        ax.plot(s.freqs[:nplot], s.amplitudes[:nplot], color="C0", lw=1)
        mean = r.props["spectrum_model"]
        freqs = s.freqs[: len(mean)]
        ax.plot(freqs, mean, color="C2")
        # ax.fill_between(freqs, mean - sfac * std, mean + sfac * std, color="C2", alpha=0.3, lw=0)
        ax.axvline(s.freqs[r.ncut - 1], ymax=0.1, color="k")
        if s.amplitudes_ref is not None:
            ax.plot(s.freqs[:nplot], s.amplitudes_ref[:nplot], **model_props)
        ax.set_xlabel(f"Frequency [1/{time_unit}]")
        ax.set_ylabel(f"Spectrum [{acf_unit} {time_unit}]")
        ax.set_title(f"ACF integral = {r.acfint:.5f} ± {r.acfint_std:.5f}")

    def plot_all_models(ax, r):
        s = r.spectrum
        for ncut, props in r.history.items():
            mean = props["spectrum_model"]
            freqs = s.freqs[: len(mean)]
            if ncut == r.ncut:
                ax.plot(freqs, mean, color="k", lw=2, zorder=2.5)
            else:
                ax.plot(freqs, mean, color="C2", lw=1, alpha=0.5)
        if s.amplitudes_ref is not None:
            nplot = min(2 * max(r.history), len(s.freqs))
            ax.plot(s.freqs[:nplot], s.amplitudes_ref[:nplot], **model_props)
        ax.set_xlabel(f"Frequency [1/{time_unit}]")
        ax.set_ylabel(f"Model Spectrum [{acf_unit} {time_unit}]")

    def plot_objective(ax, r):
        freqs = []
        objs = []
        s = r.spectrum
        for ncut, props in sorted(r.history.items()):
            freqs.append(s.freqs[ncut])
            objs.append(props["obj"])

        ax.plot(freqs, objs, color="C1", lw=1)
        ax.plot([s.freqs[r.ncut]], [r.props["obj"]], marker="o", color="k", ms=2)
        ax.set_xlabel(f"Cutoff frequency [1/{time_unit}]")
        ax.set_ylabel("Objective function")
        ncutmax = max(r.history)
        ax.set_ylim(r.props["obj"] * 1.2, ncutmax / 4)

    def plot_uncertainty(ax, r):
        freqs = []
        acfints = []
        acfint_stds = []
        s = r.spectrum
        for ncut, props in sorted(r.history.items()):
            freqs.append(s.freqs[ncut])
            acfints.append(props["pars"][:2].sum())
            acfint_stds.append(np.sqrt(props["covar"][:2, :2].sum()))
        acfints = np.array(acfints)
        acfint_stds = np.array(acfint_stds)
        ax.plot(freqs, acfints, "C3")
        ax.fill_between(
            freqs,
            acfints - sfac * acfint_stds,
            acfints + sfac * acfint_stds,
            color="C3",
            alpha=0.3,
            lw=0,
        )
        s = r.spectrum
        ax.errorbar(
            [s.freqs[r.ncut]], [r.acfint], [r.acfint_std * sfac], marker="o", ms=2, color="k"
        )
        if s.amplitudes_ref is not None:
            limit = s.amplitudes_ref[0]
            ax.axhline(limit, **model_props)
        ax.set_xlabel(f"Cutoff frequency [1/{time_unit}]")
        ax.set_ylabel(f"ACF integral [{acf_unit} {time_unit}]")

    def plot_evals(ax, r):
        freqs = []
        evals = []
        s = r.spectrum
        for ncut, props in sorted(r.history.items()):
            freqs.append(s.freqs[ncut])
            evals.append(np.linalg.eigvalsh(props["covar"]))
            if ncut == r.ncut:
                ax.plot([freqs[-1]], [evals[-1]], color="k", marker="o", ms=2, zorder=2.5)
        evals = np.array(evals)
        ax.plot(freqs, evals, color="C4")
        ax.set_xlabel(f"Cutoff frequency [1/{time_unit}]")
        ax.set_ylabel("Covariance eigenvalues")
        ax.set_yscale("log")

    def plot_uni(ax, r):
        nor = r.props["nor"]
        ax.plot((np.cumsum(nor) - nor.sum() / 2) ** 2)
        # spec = 2 * abs(np.fft.rfft(nor)[1:])**2 / len(nor)
        # ax.set_title(format(stats.chi2(2).logpdf(spec).mean(), ".3e"))
        # ax.plot(spec)
        # ax.plot(spec.real)
        # ax.plot(spec.imag)
        # print(nor.std())
        # print(spec.real.std())
        # print(spec.imag[1:].std())
        # print()
        # qgrid = (np.arange(len(uni)) + 0.5) / len(uni)
        # ax.plot(qgrid, uni)

    def plot_qq(ax):
        r0 = res[0]
        cdfs = (np.arange(len(res)) + 0.5) / len(res)
        quantiles = stats.norm().ppf(cdfs)
        limit = r0.spectrum.amplitudes_ref[0]
        normed_errors = np.array([(r.acfint - limit) / r.acfint_std for r in res])
        normed_errors.sort()
        ax.scatter(quantiles, normed_errors, c="C0", s=3)
        ax.plot([-2, 2], [-2, 2], **model_props)
        ax.set_xlabel("Normal quantiles [1]")
        ax.set_ylabel("Sorted normalized errors [1]")
        ax.set_title("QQ Plot")

    def plot_acfint_estimates(ax):
        values = np.array([r.acfint for r in res])
        errors = np.array([sfac * r.acfint_std for r in res])
        order = values.argsort()
        values = values[order]
        errors = errors[order]
        ax.errorbar(np.arange(len(res)), values, errors, fmt="o", lw=1, ms=2, ls="none")
        r0 = res[0]
        if r0.spectrum.amplitudes_ref is not None:
            limit = r0.spectrum.amplitudes_ref[0]
            ax.axhline(limit, **model_props)
        ax.set_xlabel("Rank")
        ax.set_ylabel(f"Mean and uncertainties [{acf_unit} {time_unit}]")
        ax.set_title("ACF Integral")

    with PdfPages(path_pdf) as pdf:
        for r in res:
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_spectrum(ax, r)
            pdf.savefig(fig)
            plt.close(fig)

            if len(r.history) > 1:
                fig, axs = plt.subplots(2, 2, figsize=(6, 6))
                plot_all_models(axs[0, 0], r)
                plot_objective(axs[0, 1], r)
                plot_uncertainty(axs[1, 0], r)
                # plot_evals(axs[1, 1], r)
                plot_uni(axs[1, 1], r)
                pdf.savefig(fig)
                plt.close(fig)

        if len(res) > 1:
            if res[0].spectrum.amplitudes_ref is not None:
                fig, ax = plt.subplots(figsize=(6, 6))
                plot_qq(ax)
                pdf.savefig(fig)
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 6))
            plot_acfint_estimates(ax)
            pdf.savefig(fig)
            plt.close(fig)
