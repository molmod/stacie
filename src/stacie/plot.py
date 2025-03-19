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
"""Plot various aspects of the results of the autocorrelation integral estimate."""

import re

import attrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from .estimate import Result
from .spectrum import Spectrum

__all__ = ("UnitConfig", "plot_results")


@attrs.define
class UnitConfig:
    """Unit configuration for plotting function.

    Note that values are *divided* by their units before plotting.
    """

    acint_symbol: str = attrs.field(default=r"\eta", kw_only=True)
    """The symbol used for the autocorrelation integral."""

    acint_unit_str: str = attrs.field(default="A s", kw_only=True)
    """The text used for the autocorrelation integral unit."""

    acint_unit: float = attrs.field(default=1.0, kw_only=True)
    """The unit of an autocorrelation integral."""

    acint_fmt: str = attrs.field(default=".2e", kw_only=True)
    """The format string for an autocorrelation integral."""

    freq_unit_str: str = attrs.field(default="Hz", kw_only=True)
    """The text used for the frequency unit."""

    freq_unit: float = attrs.field(default=1.0, kw_only=True)
    """The unit of a frequency."""

    time_unit_str: str = attrs.field(default="s", kw_only=True)
    """The text used for a time unit."""

    time_unit: float = attrs.field(default=1.0, kw_only=True)
    """The unit of a frequency."""

    time_fmt: str = attrs.field(default=".2e", kw_only=True)
    """The format string for a time value."""

    sfac: float = attrs.field(default=2.0)
    """The scale factor used for error bars (multiplier for sigma, standard error)."""


def fixformat(s: str) -> str:
    """Replace standard exponential notation with prettier unicode formatting."""

    def repl(match):
        factor = match.group(1)
        exp = str(int(match.group(2)))
        if exp == "0":
            return factor
        return f"${factor}\\times 10^{{{exp}}}$"

    result = re.sub(r"\b([0-9.]+)e([0-9+-]+)\b", repl, s)
    result = re.sub(r"\binf\b", "∞", result)
    return re.sub(r"\bnan\b", "?", result)


def plot_results(
    path_pdf: str, rs: Result | list[Result], uc: UnitConfig | None = None, title: str | None = None
):
    """Generate a multi-page PDF with plots of the autocorrelation integral estimation.

    Parameters
    ----------
    path_pdf
        The PDF file where all the figures are stored.
    rs
        A single ``Result`` instance or a list of them.
        If the (first) result instance has ``spectrum.amplitudes_ref`` set,
        theoretical expectations are included.
        When multiple results instances are given,
        only the first one is plotted in blue.
        All remaining ones are plotted in light grey.
    uc
        The configuration of the units used for plotting.
    """
    # Prepare results list
    if isinstance(rs, Result):
        rs = [rs]

    # Prepare units
    if uc is None:
        uc = UnitConfig()

    with PdfPages(path_pdf) as pdf:
        for r in rs:
            fig, ax = plt.subplots(figsize=(9, 6))
            plot_fitted_spectrum(ax, uc, r)
            if title is not None:
                ax.set_title(title)
            pdf.savefig(fig)
            plt.close(fig)

            if len(r.history) > 1:
                fig, axs = plt.subplots(2, 3, figsize=(9, 6))
                plot_all_models(axs[0, 0], uc, r)
                plot_criterion(axs[1, 0], uc, r)
                plot_uncertainty(axs[0, 1], uc, r)
                plot_residuals(axs[1, 1], uc, r)
                plot_evals(axs[0, 2], uc, r)
                plot_sensitivity(axs[1, 2], uc, r)
                pdf.savefig(fig)
                plt.close(fig)

        if len(rs) > 1:
            if rs[0].spectrum.amplitudes_ref is not None:
                fig, ax = plt.subplots(figsize=(9, 6))
                plot_qq(ax, uc, rs)
                pdf.savefig(fig)
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 6))
            plot_acint_estimates(ax, uc, rs)
            pdf.savefig(fig)
            plt.close(fig)


REF_PROPS = {"ls": "--", "color": "k", "alpha": 0.5}


def plot_spectrum(ax: mpl.axes.Axes, uc: UnitConfig, s: Spectrum, nplot: int | None = None):
    """Plot the empirical spectrum."""
    if nplot is None:
        nplot = s.nfreq
    ax.plot(
        s.freqs[:nplot] / uc.freq_unit,
        s.amplitudes[:nplot] / uc.acint_unit,
        color="C0",
        lw=1,
    )
    _plot_ref_spectrum(ax, uc, s, nplot)
    ax.set_xlim(left=0)
    ax.set_xlabel(f"Frequency [{uc.freq_unit_str}]")
    ax.set_ylabel(f"Spectrum [{uc.acint_unit_str}]")
    ax.set_title("Spectrum")


def _plot_ref_spectrum(ax: mpl.axes.Axes, uc: UnitConfig, s: Spectrum, nplot: int):
    """Plot the reference spectrum."""
    if s.amplitudes_ref is not None:
        ax.plot(
            s.freqs[:nplot] / uc.freq_unit,
            s.amplitudes_ref[:nplot] / uc.acint_unit,
            **REF_PROPS,
        )


FIT_LEFT_TITLE_TEMPLATE = (
    "Spectrum model {model}\n"
    "${uc.acint_symbol}$ = {acint:{uc.acint_fmt}} ± {acint_std:{uc.acint_fmt}}"
    "{acint_unit_str}"
)

FIT_RIGHT_TITLE_TEMPLATE = (
    "$\\tau_\\text{{int}}$ = {corrtime_int:{uc.time_fmt}}"
    " ± {corrtime_int_std:{uc.time_fmt}}"
    "{time_unit_str}"
)

FIT_RIGHT_TITLE_TEMPLATE_EXP = (
    "$\\tau_\\text{{exp}}$ = {corrtime_exp:{uc.time_fmt}}"
    " ± {corrtime_exp_std:{uc.time_fmt}}"
    "{time_unit_str}"
)


def plot_fitted_spectrum(ax: mpl.axes.Axes, uc: UnitConfig, r: Result):
    """Plot the fitted model spectrum."""
    nplot = 2 * r.nfit
    plot_spectrum(ax, uc, r.spectrum, nplot)
    kappas = 0.5 * r.spectrum.ndofs[: r.nfit]
    mean = r.props["thetas"] * kappas
    std = r.props["thetas"] * np.sqrt(kappas)
    freqs = r.spectrum.freqs[: r.nfit]
    ax.plot(freqs / uc.freq_unit, mean / uc.acint_unit, color="C2")
    ax.fill_between(
        freqs / uc.freq_unit,
        (mean - uc.sfac * std) / uc.acint_unit,
        (mean + uc.sfac * std) / uc.acint_unit,
        color="C2",
        alpha=0.3,
        lw=0,
    )
    ax.axvline(r.spectrum.freqs[r.nfit - 1] / uc.freq_unit, ymax=0.1, color="k")
    fields = {
        "uc": uc,
        "model": r.props["model"],
        "acint": r.acint / uc.acint_unit,
        "acint_std": r.acint_std / uc.acint_unit,
        "acint_unit_str": "" if uc.acint_unit_str == "1" else " " + uc.acint_unit_str,
        "corrtime_int": r.corrtime_int / uc.time_unit,
        "corrtime_int_std": r.corrtime_int_std / uc.time_unit,
        "time_unit_str": "" if uc.time_unit_str == "1" else " " + uc.time_unit_str,
    }
    ax.set_title("")
    ax.set_title(fixformat(FIT_LEFT_TITLE_TEMPLATE.format(**fields)), loc="left")
    if "corrtime_exp" in r.props:
        fields["corrtime_exp"] = r.props["corrtime_exp"] / uc.time_unit
        fields["corrtime_exp_std"] = r.props["corrtime_exp_std"] / uc.time_unit
        ax.set_title(
            fixformat(
                FIT_RIGHT_TITLE_TEMPLATE_EXP.format(**fields)
                + "\n"
                + FIT_RIGHT_TITLE_TEMPLATE.format(**fields)
            ),
            loc="right",
        )
    else:
        ax.set_title("\n" + fixformat(FIT_RIGHT_TITLE_TEMPLATE.format(**fields)), loc="right")


def plot_all_models(ax: mpl.axes.Axes, uc: UnitConfig, r: Result):
    """Plot all fitted model spectra (for all tested cutoffs)."""
    for nfit, props in r.history.items():
        kappas = 0.5 * r.spectrum.ndofs[:nfit]
        mean = props["thetas"] * kappas
        freqs = r.spectrum.freqs[:nfit]
        if nfit == r.nfit:
            ax.plot(freqs / uc.freq_unit, mean / uc.acint_unit, color="k", lw=2, zorder=2.5)
        else:
            ax.plot(freqs / uc.freq_unit, mean / uc.acint_unit, color="C2", lw=1, alpha=0.5)
    nplot = min(2 * max(r.history), r.spectrum.nfreq)
    _plot_ref_spectrum(ax, uc, r.spectrum, nplot)
    # Print the number of fitted model spectra in the title to show how many models were tested.
    ax.set_title(f"Model spectra ({len(r.history)} models)")
    ax.set_xlabel(f"Frequency [{uc.freq_unit_str}]")
    ax.set_ylabel(f"Model Spectrum [{uc.acint_unit_str}]")
    ax.set_xscale("log")


def plot_criterion(ax: mpl.axes.Axes, uc: UnitConfig, r: Result):
    """Plot the cutoff criterion as a function of cutoff frequency."""
    freqs = []
    criteria = []
    expected = []
    for nfit, props in sorted(r.history.items()):
        freqs.append(r.spectrum.freqs[nfit - 1])
        criteria.append(props["criterion"])
        expected.append(props.get("criterion_expected", np.nan))
    freqs = np.array(freqs)
    criteria = np.array(criteria)
    expected = np.array(expected)
    mask = np.isfinite(criteria)
    criterion_min = criteria[mask].min()
    criterion_scale = r.props.get("criterion_scale")
    if criterion_scale is None and mask.any():
        criterion_scale = np.median(criteria[mask]) - criterion_min

    if np.isfinite(expected).any():
        ax.plot(freqs / uc.freq_unit, expected, color="C1", lw=1, alpha=0.5, ls="--")
    ax.plot(freqs / uc.freq_unit, criteria, color="C1", lw=1)
    ax.axvline(r.spectrum.freqs[r.nfit - 1] / uc.freq_unit, ymax=0.1, color="k")
    ax.axhline(0, **REF_PROPS)
    ax.set_xlabel(f"Cutoff frequency [{uc.freq_unit_str}]")
    ax.set_ylabel("Criterion")
    ax.set_title("Cutoff criterion")
    ax.set_xscale("log")
    if criterion_scale is not None:
        ax.set_ylim(criterion_min - 0.2 * criterion_scale, criterion_min + 2.5 * criterion_scale)


def plot_uncertainty(ax: mpl.axes.Axes, uc: UnitConfig, r: Result):
    """Plot the autocorrelation integral and uncertainty as a function fo cutoff frequency."""
    freqs = []
    acints = []
    acint_stds = []
    s = r.spectrum
    for nfit, props in sorted(r.history.items()):
        freqs.append(r.spectrum.freqs[nfit - 1])
        acints.append(props["acint"])
        acint_stds.append(props["acint_std"])
    freqs = np.array(freqs)
    acints = np.array(acints)
    acint_stds = np.array(acint_stds)

    ax.plot(freqs / uc.freq_unit, acints / uc.acint_unit, "C3")
    ax.fill_between(
        freqs / uc.freq_unit,
        (acints - uc.sfac * acint_stds) / uc.acint_unit,
        (acints + uc.sfac * acint_stds) / uc.acint_unit,
        color="C3",
        alpha=0.3,
        lw=0,
    )
    s = r.spectrum
    ax.errorbar(
        [r.spectrum.freqs[r.nfit - 1] / uc.freq_unit],
        [r.acint / uc.acint_unit],
        [r.acint_std * uc.sfac / uc.acint_unit],
        marker="o",
        ms=2,
        color="k",
    )
    if s.amplitudes_ref is not None:
        limit = s.amplitudes_ref[0]
        ax.axhline(limit / uc.acint_unit, **REF_PROPS)
    ax.set_xlabel(f"Cutoff frequency [{uc.freq_unit_str}]")
    ax.set_ylabel(f"Autocorrelation integral [{uc.acint_unit_str}]")
    ax.set_xscale("log")


def plot_evals(ax: mpl.axes.Axes, uc: UnitConfig, r: Result):
    """Plot the eigenvalues of the Hessian as a function of the cutoff frequency."""
    freqs = []
    evals = []
    for nfit, props in sorted(r.history.items()):
        freqs.append(r.spectrum.freqs[nfit - 1])
        evals.append(1 / props["cost_hess_evals"])
        if nfit == r.nfit:
            ax.plot([freqs[-1]], [evals[-1]], color="k", marker="o", ms=2, zorder=2.5)
    freqs = np.array(freqs)
    evals = np.array(evals)

    ax.plot(freqs / uc.freq_unit, evals, color="C4")
    ax.set_xlabel(f"Cutoff frequency [{uc.freq_unit_str}]")
    ax.set_ylabel("Covariance eigenvalues")
    ax.set_yscale("log")
    ax.set_xscale("log")


def plot_residuals(ax: mpl.axes.Axes, uc: UnitConfig, r: Result):
    """Plot the normalized residuals between the model and empirical spectra."""
    amplitudes = r.spectrum.amplitudes[: r.nfit]
    kappas = 0.5 * r.spectrum.ndofs[: r.nfit]
    thetas = r.props["thetas"]
    residuals = (amplitudes / thetas - kappas) / np.sqrt(kappas)
    with np.errstate(invalid="ignore"):
        ax.plot(r.spectrum.freqs[: r.nfit] / uc.freq_unit, residuals, color="C0")
        ax.axhline(0, ls="--", lw=1.0, color="k")
    ax.set_title("Normalized residuals")
    ax.set_xlabel(f"Frequency [{uc.freq_unit_str}]")
    ax.set_ylabel("Residual [1]")


def plot_sensitivity(ax: mpl.axes.Axes, uc: UnitConfig, r: Result):
    """Plot the sensitivity of the autocorrelation integral estimate to spectrum amplitudes."""
    ax.plot(r.spectrum.freqs[: r.nfit] / uc.freq_unit, r.props["acint_sensitivity"], color="C5")
    ax.axhline(0, ls="--", lw=1.0, color="k")
    ax.set_title("Sensitivity of the autocorrelation integral to spectrum amplitudes", wrap=True)
    ax.set_xlabel(f"Frequency [{uc.freq_unit_str}]")
    ax.set_ylabel("Sensitivity [1]")


def plot_qq(ax: mpl.axes.Axes, uc: UnitConfig, rs: list[Result]):
    """Make a qq-plot between the predicted and expected distribution of AC integral estimates.

    This plot function assumes the true integral is known.
    """
    cdfs = (np.arange(len(rs)) + 0.5) / len(rs)
    quantiles = stats.norm().ppf(cdfs)
    limit = rs[0].spectrum.amplitudes_ref[0]
    normed_errors = np.array([(r.acint - limit) / r.acint_std for r in rs])
    normed_errors.sort()
    distance = abs(quantiles - normed_errors).mean()
    ax.scatter(quantiles, normed_errors, c="C0", s=3)
    ax.plot([-2, 2], [-2, 2], **REF_PROPS)
    ax.set_xlabel("Normal quantiles [1]")
    ax.set_ylabel("Sorted normalized errors [1]")
    ax.set_title(f"QQ Plot (Wasserstein Distance = {distance:.4f})")


RELERR_TEMPLATE = """\
MRE = {mre:.1f} %
RMSRE = {rmsre:.1f} %
RMSRF = {rmsrf:.1f} %
RMSPRE = {rmspre:.1f} %
"""


def rms(x):
    return np.sqrt((x**2).mean())


def plot_acint_estimates(ax: mpl.axes.Axes, uc: UnitConfig, rs: list[Result]):
    """Plot the sorted autocorrelation integral estimates and their uncertainties."""
    values = np.array([r.acint for r in rs])
    errors = np.array([r.acint_std for r in rs])
    order = values.argsort()
    values = values[order]
    errors = errors[order]
    ax.errorbar(
        np.arange(len(rs)),
        values / uc.acint_unit,
        uc.sfac * errors,
        fmt="o",
        lw=1,
        ms=2,
        ls="none",
    )
    if rs[0].spectrum.amplitudes_ref is not None:
        limit = rs[0].spectrum.amplitudes_ref[0]
        ax.axhline(limit / uc.acint_unit, **REF_PROPS)
        relative_errors = 100 * (values - limit) / values
        mre = relative_errors.mean()
        ax.text(
            0.05,
            0.95,
            RELERR_TEMPLATE.format(
                mre=mre,
                rmsre=rms(relative_errors),
                rmsrf=rms(relative_errors - mre),
                rmspre=rms(errors / values) * 100,
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    ax.set_xlabel("Rank")
    ax.set_ylabel(f"Mean and uncertainty [{uc.acint_unit_str}]")
    ax.set_title("Autocorrelation Integral")
