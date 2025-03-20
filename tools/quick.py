#!/usr/bin/env python3
"""A minimalistic example demonstrating quick use of Stacie."""

import matplotlib as mpl
import numpy as np

from stacie import ChebyshevModel, compute_spectrum, estimate_acint, plot_results, summarize_results

# Generate input sequences with a simple Markov process.
nseq = 100
sequences = [np.zeros(nseq)]
nstep = 8192
rng = np.random.default_rng(4)
for _ in range(nstep):
    step = sequences[-1] * 0.9
    step += rng.normal(0, 0.1, nseq)
    sequences.append(step)

# Bring the sequences into the right shape for the spectrum computation.
sequences = np.array(sequences).T

# Estimate the autocorrelation integral, print and plot the results.
spectrum = compute_spectrum(sequences)
result = estimate_acint(spectrum, ChebyshevModel(2), verbose=True)
print()
print(summarize_results(result))
mpl.rc_file("../docs/source/examples/matplotlibrc")
plot_results("quick.pdf", result)
