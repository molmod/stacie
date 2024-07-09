#!/usr/bin/env python
"""Generate synthetic signals and store their spectra in NPZ files for testing."""

import numpy as np
from celerite2 import GaussianProcess, terms
from stacie.spectrum import Spectrum, prepare_acfint
from stacie.zarr import dump


def main():
    dump(generate_white(1, 2000, 400), "../tests/inputs/spectrum_white1.zip")
    dump(generate_white(2, 2000, 400), "../tests/inputs/spectrum_white2.zip")
    dump(generate_double(1, 2000, 400), "../tests/inputs/spectrum_double1.zip")
    dump(generate_double(2, 2000, 400), "../tests/inputs/spectrum_double2.zip")


def generate_white(seed, nstep, nindep) -> Spectrum:
    "Generate an averaged white-noise spectrum"
    rng = np.random.default_rng(seed)
    sequences = rng.normal(0, 16, (nindep, nstep))
    spectrum = prepare_acfint(sequences)
    spectrum.amplitudes_ref = np.full_like(spectrum.freqs, 0.5 * 16**2)
    return spectrum


def generate_double(seed, nstep, nindep) -> Spectrum:
    """Generate time-correlated data with a double stochastic harmonic oscillator."""
    kernel = terms.SHOTerm(S0=50.0, w0=0.1, Q=1.0) + terms.SHOTerm(S0=100.0, w0=0.1, Q=0.1)
    gp = GaussianProcess(kernel)
    time = np.arange(nstep, dtype=float)
    gp.compute(time)
    # Celerite does not support the new RNG interface of NumPy (yet).
    np.random.seed(seed)  # noqa: NPY002
    sequences = gp.sample(size=nindep)
    spectrum = prepare_acfint(sequences)
    freqs = np.fft.rfftfreq(nstep)
    omegas = 2 * np.pi * freqs
    spectrum.amplitudes_ref = gp.kernel.get_psd(omegas) * np.sqrt(np.pi / 2)
    return spectrum


if __name__ == "__main__":
    main()
