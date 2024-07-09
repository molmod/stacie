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

import attrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from .estimate import Result

__all__ = ("plot", "UnitConfig")


@attrs.define
class UnitConfig:
    """Unit configuration for plotting function.

    Note that values are *divided* by their units before plotting.
    """

    acfint_unit_str: str = attrs.field(default="A s")
    """The text used to label an ACF integral."""

    acfint_unit: float = attrs.field(default=1.0)
    """The unit of an ACF integral."""

    acfint_fmt: str = attrs.field(default=".1f")
    """The format string for an ACF integral."""

    freq_unit_str: str = attrs.field(default="Hz")
    """The text used to label a frequency value."""

    freq_unit: float = attrs.field(default=1.0)
    """The unit of a frequency."""

    time_unit_str: str = attrs.field(default="s")
    """The text used to label a time value."""

    time_unit: float = attrs.field(default=1.0)
    """The unit of a frequency."""

    time_fmt: str = attrs.field(default=".1f")
    """The format string for a time value."""

    sfac: float = attrs.field(default=2.0)
    """The scale factor used for error bars (multiplier for sigma, standard error)."""


def plot(path_pdf: str, res: Result | list[Result], uc: UnitConfig | None = None):
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

    # Prepare units
    if uc is None:
        uc = UnitConfig()

    model_props = {"ls": "--", "color": "k", "alpha": 0.5}

    def plot_spectrum(ax, r: Result):
        nplot = 2 * r.ncut
        s = r.spectrum
        ax.plot(
            s.freqs[:nplot] / uc.freq_unit,
            s.amplitudes[:nplot] / uc.acfint_unit,
            color="C0",
            lw=1,
        )
        mean = r.props["amplitudes_model"]
        std = r.props["amplitudes_std_model"]
        freqs = s.freqs[: len(mean)]
        ax.plot(freqs / uc.freq_unit, mean / uc.acfint_unit, color="C2")
        ax.fill_between(
            freqs / uc.freq_unit,
            (mean - uc.sfac * std) / uc.acfint_unit,
            (mean + uc.sfac * std) / uc.acfint_unit,
            color="C2",
            alpha=0.3,
            lw=0,
        )
        ax.axvline(s.freqs[r.ncut - 1] / uc.freq_unit, ymax=0.1, color="k")
        if s.amplitudes_ref is not None:
            ax.plot(
                s.freqs[:nplot] / uc.freq_unit,
                s.amplitudes_ref[:nplot] / uc.acfint_unit,
                **model_props,
            )
        ax.set_xlabel(f"Frequency [{uc.freq_unit_str}]")
        ax.set_ylabel(f"Spectrum [{uc.acfint_unit_str}]")
        ax.set_title(
            f"{r.props['model']} // "
            f"ACF integral = {r.props['acfint'] / uc.acfint_unit:{uc.acfint_fmt}} "
            f"± {r.props['acfint_std'] / uc.acfint_unit:{uc.acfint_fmt}} "
            f"{uc.acfint_unit_str} // "
            f"Corrtime tail = {r.props['corrtime_tail'] / uc.time_unit:{uc.time_fmt}} "
            f"± {r.props['corrtime_tail_std'] / uc.time_unit:{uc.time_fmt}} " + uc.time_unit_str
        )

    def plot_all_models(ax, r):
        s = r.spectrum
        for ncut, props in r.history.items():
            mean = props["amplitudes_model"]
            freqs = s.freqs[: len(mean)]
            if ncut == r.ncut:
                ax.plot(freqs / uc.freq_unit, mean / uc.acfint_unit, color="k", lw=2, zorder=2.5)
            else:
                ax.plot(freqs / uc.freq_unit, mean / uc.acfint_unit, color="C2", lw=1, alpha=0.5)
        if s.amplitudes_ref is not None:
            nplot = min(2 * max(r.history), len(s.freqs))
            ax.plot(
                s.freqs[:nplot] / uc.freq_unit,
                s.amplitudes_ref[:nplot] / uc.acfint_unit,
                **model_props,
            )
        ax.set_xlabel(f"Frequency [{uc.freq_unit_str}]")
        ax.set_ylabel(f"Model Spectrum [{uc.acfint_unit_str}]")

    def plot_objective(ax, r):
        freqs = []
        objs = []
        s = r.spectrum
        for ncut, props in sorted(r.history.items()):
            freqs.append(s.freqs[ncut])
            objs.append(props["obj"])
        freqs = np.array(freqs)

        ax.plot(freqs / uc.freq_unit, objs, color="C1", lw=1)
        ax.plot([s.freqs[r.ncut] / uc.freq_unit], [r.props["obj"]], marker="o", color="k", ms=2)
        ax.set_xlabel(f"Cutoff frequency [{uc.freq_unit_str}]")
        ax.set_ylabel("Objective function [1]")
        ncutmax = max(r.history)
        if np.isfinite(r.props["obj"]):
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
        freqs = np.array(freqs)
        acfints = np.array(acfints)
        acfint_stds = np.array(acfint_stds)

        ax.plot(freqs / uc.freq_unit, acfints / uc.acfint_unit, "C3")
        ax.fill_between(
            freqs / uc.freq_unit,
            (acfints - uc.sfac * acfint_stds) / uc.acfint_unit,
            (acfints + uc.sfac * acfint_stds) / uc.acfint_unit,
            color="C3",
            alpha=0.3,
            lw=0,
        )
        s = r.spectrum
        ax.errorbar(
            [s.freqs[r.ncut] / uc.freq_unit],
            [r.props["acfint"] / uc.acfint_unit],
            [r.props["acfint_std"] * uc.sfac / uc.acfint_unit],
            marker="o",
            ms=2,
            color="k",
        )
        if s.amplitudes_ref is not None:
            limit = s.amplitudes_ref[0]
            ax.axhline(limit / uc.acfint_unit, **model_props)
        ax.set_xlabel(f"Cutoff frequency [{uc.freq_unit_str}]")
        ax.set_ylabel(f"ACF integral [{uc.acfint_unit_str}]")

    def plot_evals(ax, r):
        freqs = []
        evals = []
        s = r.spectrum
        for ncut, props in sorted(r.history.items()):
            freqs.append(s.freqs[ncut])
            evals.append(np.linalg.eigvalsh(props["covar"]))
            if ncut == r.ncut:
                ax.plot([freqs[-1]], [evals[-1]], color="k", marker="o", ms=2, zorder=2.5)
        freqs = np.array(freqs)
        evals = np.array(evals)

        ax.plot(freqs / uc.freq_unit, evals, color="C4")
        ax.set_xlabel(f"Cutoff frequency [{uc.freq_unit_str}]")
        ax.set_ylabel("Covariance eigenvalues")
        ax.set_yscale("log")

    def plot_nor(ax, r):
        nor = r.props["nor"]
        with np.errstate(invalid="ignore"):
            ax.plot(np.cumsum(nor) - np.sum(nor) / 2)
        ax.set_title("symcu normalized residuals")
        ax.set_xlabel("index")
        ax.set_ylabel("symcu")

    def plot_qq(ax):
        r0 = res[0]
        cdfs = (np.arange(len(res)) + 0.5) / len(res)
        quantiles = stats.norm().ppf(cdfs)
        limit = r0.spectrum.amplitudes_ref[0]
        normed_errors = np.array([(r.props["acfint"] - limit) / r.props["acfint_std"] for r in res])
        normed_errors.sort()
        ax.scatter(quantiles, normed_errors, c="C0", s=3)
        ax.plot([-2, 2], [-2, 2], **model_props)
        ax.set_xlabel("Normal quantiles [1]")
        ax.set_ylabel("Sorted normalized errors [1]")
        ax.set_title("QQ Plot")

    def plot_acfint_estimates(ax):
        values = np.array([r.props["acfint"] for r in res])
        errors = np.array([uc.sfac * r.props["acfint_std"] for r in res])
        order = values.argsort()
        values = values[order]
        errors = errors[order]
        ax.errorbar(
            np.arange(len(res)),
            values / uc.acfint_unit,
            errors,
            fmt="o",
            lw=1,
            ms=2,
            ls="none",
        )
        r0 = res[0]
        if r0.spectrum.amplitudes_ref is not None:
            limit = r0.spectrum.amplitudes_ref[0]
            ax.axhline(limit / uc.acfint_unit, **model_props)
        ax.set_xlabel("Rank")
        ax.set_ylabel(f"Mean and uncertainty [{uc.acfint_unit_str}]")
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
                plot_nor(axs[1, 1], r)
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
