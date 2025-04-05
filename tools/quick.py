#!/usr/bin/env python3
"""A minimalistic example demonstrating quick use of Stacie."""

import matplotlib as mpl
import numpy as np

from stacie import (
    PolynomialModel,
    compute_spectrum,
    estimate_acint,
    plot_results,
    summarize_results,
)

# Generate 64 input sequences with 8192 steps, using a simple Markov process.
# The autocorrelation integral is 1.0
# The integrated correlation time is 16.0
nseq = 64
nstep = 8192
alpha = 31 / 33
beta = np.sqrt(8 / 1089)
rng = np.random.default_rng(0)
sequences = np.zeros((nseq, nstep))
for i in range(1, nstep):
    sequences[:, i] = alpha * sequences[:, i - 1] + rng.normal(0, beta, nseq)

# Estimate the autocorrelation integral, print and plot the results.
spectrum = compute_spectrum(sequences)
result = estimate_acint(spectrum, PolynomialModel(2, even=True), verbose=True)
print(summarize_results(result))
mpl.rc_file("../docs/source/examples/matplotlibrc")
plot_results("quick.pdf", result)
