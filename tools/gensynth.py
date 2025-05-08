#!/usr/bin/env python3
"""Generate synthetic signals and store their spectra as compressed messagepack files for testing.

Four spectra are generated:

- A `white` noise spectrum.
- A `pure` Lorentzian (without white noise added).
- A `broad` Lorentzian (with white noise added).
- A `double` Lorentzian (without white noise added).

All spectra are generated with 8192 time steps and 256 independent realizations.
The ground truth of the spectrum is stored in `spectrum.amplitudes_ref`.
The expected value of the autocorrelation integral is always 1.
"""

import attrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from stacie.plot import UnitConfig, plot_spectrum
from stacie.spectrum import Spectrum, compute_spectrum
from stacie.synthetic import generate


def generate_lorentzian(
    name: str, seed: int, nseq: int, nstep: int, awhite: float, lterms: list
) -> Spectrum:
    """Generate a white-noise + Lorentzian(s) spectrum
    Parameters
    ----------
    name
        Name of the spectrum.
    seed
        Seed for the random number generator.
    nseq
        Number of independent realizations.
    nstep
        Number of time steps.
    awhite
        Amplitude of the white noise.
    lterms
        List of (amplitude, time constant) tuples for Lorentzian terms.
        Use an empty list for white noise.

    Returns
    -------
        The generated spectrum.
    """
    psd = np.full(nstep // 2 + 1, awhite)
    freqs = np.fft.rfftfreq(nstep)
    for amplitude, tau in lterms:
        omegas = 2 * np.pi * freqs
        psd += amplitude / (1 + (tau * omegas) ** 2)
    if abs(psd[0] - 1.0) > 1e-6:
        raise ValueError("DC component is not unity")
    rng = np.random.default_rng(seed)
    sequences = generate(psd, 1.0, nseq, nstep, rng)
    spectrum = compute_spectrum(sequences, prefactors=2.0)
    spectrum.amplitudes_ref = psd
    np.savez_compressed("../tests/inputs/spectrum_" + name, **attrs.asdict(spectrum))

    mpl.rc_file("../docs/source/examples/matplotlibrc")
    fig, ax = plt.subplots()
    uc = UnitConfig()
    plot_spectrum(ax, uc, spectrum)
    if name != "white":
        ax.set_title(f"Spectrum ({name} Lorentzian)")
    else:
        ax.set_title(f"Spectrum ({name})")

    info = f"No. of sequences: {nseq}\nNo. of steps: {nstep}"
    ax.text(
        0.9,
        0.9,
        info,
        ha="center",
        va="top",
        transform=ax.transAxes,
        linespacing=1.5,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "green",
            "linewidth": 1.5,
        },
        fontsize=10,
    )
    ax.set_xlim(0, 0.1)
    fig.savefig("spectrum_" + name + ".pdf")


def main():
    generate_lorentzian("white", 1, 256, 8192, 1.0, [])
    generate_lorentzian("pure", 2, 256, 8192, 0.0, [(1.0, 40.0)])
    generate_lorentzian("broad", 3, 256, 8192, 0.2, [(0.8, 4.0)])
    generate_lorentzian("double", 4, 256, 8192, 0.0, [(0.1, 4.0), (0.9, 40.0)])


if __name__ == "__main__":
    main()
