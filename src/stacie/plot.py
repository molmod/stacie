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
"""Plot various aspects of the results of the autocorrelation integral estimate."""

import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import NDArray
from scipy import stats

from .estimate import Result
from .spectrum import Spectrum
from .utils import UnitConfig, label_unit

__all__ = ("plot_results",)


def fixformat(s: str) -> str:
    """Replace standard scientific notation with prettier unicode formatting."""

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
    path_pdf: str,
    rs: Result | list[Result],
    uc: UnitConfig | None = None,
    *,
    figsize: tuple = (7.5, 4.21875),
    legend: bool = True,
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
    figsize
        The figure size tuple for matplotlib
    """
    # Prepare results list
    if isinstance(rs, Result):
        rs = [rs]

    # Prepare units
    if uc is None:
        uc = UnitConfig()

    with PdfPages(path_pdf) as pdf:
        for r in rs:
            fig, ax = plt.subplots(figsize=figsize)
            plot_fitted_spectrum(ax, uc, r, legend=legend)
            pdf.savefig(fig)
            plt.close(fig)

            if len(r.history) > 1:
                fig, axs = plt.subplots(2, 2, figsize=figsize)
                plot_extras(axs, uc, r)
                pdf.savefig(fig)
                plt.close(fig)

        if len(rs) > 1:
            if rs[0].spectrum.amplitudes_ref is not None:
                fig, ax = plt.subplots(figsize=figsize)
                plot_qq(ax, uc, rs)
                pdf.savefig(fig)
                plt.close(fig)

            fig, ax = plt.subplots(figsize=figsize)
            plot_acint_estimates(ax, uc, rs)
            pdf.savefig(fig)
            plt.close(fig)


REF_PROPS = {"ls": "--", "color": "k", "alpha": 0.5}


def plot_spectrum(ax: mpl.axes.Axes, uc: UnitConfig, s: Spectrum, nplot: int | None = None):
    """Plot the empirical spectrum."""
    if nplot is None or nplot > s.nfreq:
        nplot = s.nfreq
    ax.plot(
        s.freqs[:nplot] / uc.freq_unit,
        s.amplitudes[:nplot] / uc.acint_unit,
        "C0.",
        mew=0,
    )
    _plot_ref_spectrum(ax, uc, s, nplot)
    fmax = s.freqs[:nplot].max() / uc.freq_unit
    ax.set_xlim(-0.01 * fmax, fmax * 1.01)
    ax.set_xlabel(label_unit("Frequency", uc.freq_unit_str))
    ax.set_ylabel(label_unit("Amplitude", uc.acint_unit_str))
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
    "Model {model} \n"
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


def plot_fitted_spectrum(ax: mpl.axes.Axes, uc: UnitConfig, r: Result, *, legend: bool = True):
    """Plot the fitted model spectrum."""
    freqs = r.spectrum.freqs[: r.ncut]
    # Show fitting weight on top of the spectrum
    ax2 = ax.twinx()
    ax2.plot(freqs / uc.freq_unit, r.props["weights"], color="C3", ls=":")
    ax2.set_ylabel("Fitting weight", color="C3")
    ax2.set_ylim(-0.02, 1.02)
    ax2.spines["right"].set_color("red")
    ax2.spines["right"].set_visible(True)
    ax2.tick_params(axis="y", colors="red")

    # The empirical spectrum.
    plot_spectrum(ax, uc, r.spectrum, r.ncut)

    # Model spectrum.
    neff = int(np.ceil(r.neff))
    freqs = r.spectrum.freqs[:neff]
    alphas = 0.5 * r.spectrum.ndofs[:neff]
    mean = r.props["amplitudes_model"][0][:neff]
    std_fit = np.sqrt(
        np.einsum(
            "ij,ik,jk->k",
            r.props["pars_covar"],
            r.props["amplitudes_model"][1][:, :neff],
            r.props["amplitudes_model"][1][:, :neff],
        )
    )
    ax.plot(freqs / uc.freq_unit, mean / uc.acint_unit, color="C2")
    ax.plot(
        freqs / uc.freq_unit,
        stats.norm.ppf(uc.clb, mean, std_fit) / uc.acint_unit,
        color="C2",
        ls="--",
    )
    ax.plot(
        freqs / uc.freq_unit,
        stats.norm.ppf(uc.cub, mean, std_fit) / uc.acint_unit,
        color="C2",
        ls="--",
        label=f"{uc.clevel:.0%} CI fitted model",
    )
    ax.fill_between(
        freqs / uc.freq_unit,
        stats.gamma.ppf(uc.clb, alphas, scale=mean / alphas) / uc.acint_unit,
        stats.gamma.ppf(uc.cub, alphas, scale=mean / alphas) / uc.acint_unit,
        color="C2",
        alpha=0.3,
        lw=0,
        label=f"{uc.clevel:.0%} CI sampling PSD",
    )
    ax.axvline(r.fcut / uc.freq_unit, ymax=0.1, color="k")
    ax.set_ylim(top=r.spectrum.amplitudes[: r.ncut].max() / uc.acint_unit * 1.1)
    if legend:
        ax.legend(loc="best")
    # Info in title
    fields = {
        "uc": uc,
        "model": r.model.name,
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
        fields["corrtime_exp"] = r.corrtime_exp / uc.time_unit
        fields["corrtime_exp_std"] = r.corrtime_exp_std / uc.time_unit
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


def plot_extras(axs: NDArray[mpl.axes.Axes], uc: UnitConfig, r: Result):
    plot_cutoff_weight(axs[0, 0], uc, r)
    plot_sanity(axs[1, 0], uc, r)
    axs[0, 0].sharex(axs[1, 0])
    axs[0, 0].set_xlabel(None)
    plot_uncertainty(axs[0, 1], uc, r)
    plot_evals(axs[1, 1], uc, r)
    axs[0, 1].sharex(axs[1, 1])
    axs[0, 1].set_xlabel(None)


def plot_cutoff_weight(ax: mpl.axes.Axes, uc: UnitConfig, r: Result):
    """Plot the cutoff criterion as a function of cutoff frequency."""
    fcuts = np.array([props["fcut"] for props in r.history])
    criteria = np.array([props["criterion"] for props in r.history])
    criteria -= criteria.min()
    probs = np.exp(-criteria)
    probs /= probs.sum()
    ax.plot(fcuts / uc.freq_unit, probs, color="C1")
    ax.set_xscale("log")
    ax.set_xlabel(label_unit("Cutoff frequency", uc.freq_unit_str))
    ax.set_ylabel(f"{r.cutoff_criterion.name} weight")


def plot_sanity(ax: mpl.axes.Axes, uc: UnitConfig, r: Result):
    fcuts = np.array([props["fcut"] for props in r.history])
    low = -1
    high = 4
    for key, color in ("cost", "C0"), ("criterion", "C2"):
        zscore = np.array([props[f"{key}_zscore"] for props in r.history])
        ax.plot(fcuts / uc.freq_unit, zscore, color=color, label=key.title())
        low = min(low, zscore.min())
        high = max(high, r.props[f"{key}_zscore"] * 1.2)
    ax.set_xscale("log")
    ax.set_xlabel(label_unit("Cutoff frequency", uc.freq_unit_str))
    ax.set_ylabel("Z-score")
    ax.axhline(0, color="k", ls="--")
    ax.axhline(2, color="k", ls="--")
    ax.set_ylim(low - 0.2, high)
    ax.legend()


def plot_uncertainty(ax: mpl.axes.Axes, uc: UnitConfig, r: Result):
    """Plot the autocorrelation integral and uncertainty as a function fo cutoff frequency."""
    fcuts = []
    acints = []
    acint_stds = []
    for props in r.history:
        fcuts.append(props["fcut"])
        acints.append(props["acint"])
        acint_stds.append(props["acint_std"])
    fcuts = np.array(fcuts)
    acints = np.array(acints)
    acint_stds = np.array(acint_stds)

    ax.plot(fcuts / uc.freq_unit, acints / uc.acint_unit, "C3")
    ax.fill_between(
        fcuts / uc.freq_unit,
        stats.norm.ppf(uc.clb, acints, acint_stds) / uc.acint_unit,
        stats.norm.ppf(uc.cub, acints, acint_stds) / uc.acint_unit,
        color="C3",
        alpha=0.3,
        lw=0,
    )
    fcut_weights = np.array([props["fcut_weight"] for props in r.history])
    fcut_weights /= fcut_weights.max()
    mask = fcut_weights > 0.01
    ax.scatter(
        fcuts[mask] / uc.freq_unit,
        acints[mask] / uc.acint_unit,
        s=30,
        c=fcut_weights[mask],
        marker="o",
        linewidth=0,
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    if r.spectrum.amplitudes_ref is not None:
        limit = r.spectrum.amplitudes_ref[0]
        ax.axhline(limit / uc.acint_unit, **REF_PROPS)
    ax.set_xscale("log")
    ax.set_xlabel(label_unit("Cutoff frequency", uc.freq_unit_str))
    ax.set_ylabel(label_unit(f"${uc.acint_symbol}$ ({uc.clevel:.0%} CI)", uc.acint_unit_str))


def plot_evals(ax: mpl.axes.Axes, uc: UnitConfig, r: Result):
    """Plot the eigenvalues of the Hessian as a function of the cutoff frequency."""
    fcuts = np.array([props["fcut"] for props in r.history])
    all_evals = np.array([props["cost_hess_rescaled_evals"] for props in r.history])
    fcut_weights = np.array([props["fcut_weight"] for props in r.history])
    fcut_weights /= fcut_weights.max()

    ax.plot(fcuts / uc.freq_unit, all_evals, color="C4")
    mask = fcut_weights > 0.01
    for evals in all_evals.T:
        ax.scatter(
            fcuts[mask] / uc.freq_unit,
            evals[mask],
            s=30,
            c=fcut_weights[mask],
            marker="o",
            linewidth=0,
            cmap="Greys",
            vmin=0,
            vmax=1,
        )
    ax.set_xscale("log")
    ax.set_xlabel(label_unit("Cutoff frequency", uc.freq_unit_str))
    ax.set_ylabel("Hessian Eigenvalues")
    ax.set_yscale("log")


def plot_all_models(ax: mpl.axes.Axes, uc: UnitConfig, r: Result):
    """Plot all fitted model spectra (for all tested cutoffs)."""
    fcut_weights = np.array([props["fcut_weight"] for props in r.history])
    fcut_weights /= fcut_weights.max()
    nplot = 0
    for i, props in enumerate(r.history):
        ncut = props["ncut"]
        nplot = max(nplot, ncut)
        mean = props["amplitudes_model"][0]
        freqs = r.spectrum.freqs[:ncut]
        if fcut_weights[i] > 0.01:
            plot_kwargs = {"alpha": fcut_weights[i], "zorder": 2.5, "color": "C2"}
        else:
            plot_kwargs = {"alpha": 0.1, "zorder": 0.5, "color": "k"}
        ax.plot(freqs / uc.freq_unit, mean / uc.acint_unit, **plot_kwargs)
    _plot_ref_spectrum(ax, uc, r.spectrum, nplot)
    # Print the number of fitted model spectra in the title to show how many models were tested.
    ax.set_title(f"Model spectra ({len(r.history)} fits)", wrap=True)
    ax.set_xlabel(label_unit("Frequency", uc.freq_unit_str))
    ax.set_ylabel(label_unit("Amplitude", uc.acint_unit_str))
    ax.set_ylim(bottom=0)


def plot_qq(ax: mpl.axes.Axes, uc: UnitConfig, rs: list[Result]):
    """Make a qq-plot between the predicted and expected distribution of AC integral estimates.

    This plot function assumes the true integral is known.
    """
    cdfs = (np.arange(len(rs)) + 0.5) / len(rs)
    quantiles = stats.norm.ppf(cdfs)
    limit = rs[0].spectrum.amplitudes_ref[0]
    normed_errors = np.array([(r.acint - limit) / r.acint_std for r in rs])
    normed_errors.sort()
    distance = abs(quantiles - normed_errors).mean()
    ax.scatter(quantiles, normed_errors, c="C0", s=3)
    ax.plot([-2, 2], [-2, 2], **REF_PROPS)
    ax.set_xlabel("Normal quantiles")
    ax.set_ylabel("Sorted normalized errors")
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
    stds = np.array([r.acint_std for r in rs])
    order = values.argsort()
    values = values[order]
    stds = stds[order]
    ax.errorbar(
        np.arange(len(rs)),
        values / uc.acint_unit,
        [
            (values - stats.norm.ppf(uc.clb, values, stds)) / uc.acint_unit,
            (stats.norm.ppf(uc.cub, values, stds) - values) / uc.acint_unit,
        ],
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
                rmspre=rms(stds / values) * 100,
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    ax.set_xlabel("Rank")
    ax.set_ylabel(label_unit("Mean and uncertainty", uc.acint_unit_str))
    ax.set_title("Autocorrelation Integral")
