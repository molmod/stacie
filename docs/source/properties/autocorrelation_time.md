# Integrated and Exponential Autocorrelation Time

## Definitions

There are two definitions of the autocorrelation time {cite:p}`sokal_1997_monte`:

1. The *integrated* autocorrelation time is derived from the autocorrelation integral:

    $$
        \tau_\text{int}
        = \frac{\int_{-\infty}^{+\infty} c(\Delta_t)\,\mathrm{d}\Delta_t}{2 c(0)}
        = \frac{\mathcal{I}}{F c(0)}
    $$

    where $c(\Delta_t)$ is the autocorrelation function,
    $\mathcal{I}$ is the ACF defined with STACIE's conventions,
    and $F$ is the prefactor of the autocorrelation integral,
    introduced in the [overview of the autocorrelation integral](../theory/overview.md).

2. The *exponential* autocorrelation time is defined as
   the limit of the exponential decay rate of the autocorrelation function.
   In STACIE's notation, this means that for large $\Delta_t$, we have:

    $$
        c(\Delta_t) \propto \exp\left(-\frac{|\Delta_t|}{\tau_\text{exp}}\right)
    $$

    The exponential autocorrelation time characterizes the slowest mode in the input.
    The parameter $\tau_\text{exp}$ can be estimated with the
    [Pade model](#section-pade-target).

Both correlation times are the same if the autocorrelation is nothing more than
a two-sided exponentially decaying function:

$$
    c(\Delta_t) = c_0 \exp\left(-\frac{|\Delta_t|}{\tau_\text{exp}}\right)
$$

In practice, however, the two correlation times may differ.
This can happen if the input sequences
are a superposition of signals with different relaxation times,
or when they contain non-diffusive contributions such as oscillations at certain frequencies.
It is even not guaranteed that the exponential autocorrelation time is always well-defined,
e.g., when the ACF decays as a power law.

## Which Definition Should I Use?

There is no right or wrong.
Both definitions are useful and relevant for different applications.

1. The integrated correlation time is related to [the variance of the mean
   of a time-correlated sequence](error_estimates.md):

    $$
        \var[\hat{x}_\text{av}] = \frac{\var[\hat{x}_n]}{N} \frac{2\tau_\text{int}}{h}
    $$

    The first factor is the "naive" variance of the mean,
    assuming that all $N$ inputs are uncorrelated.
    The second factor corrects for the presence of time correlations
    and is called the statistical inefficiency
    {cite:p}`allen_2017_computer,friedberg_1970_test`:

    $$
        s = \frac{2\tau_\text{int}}{h}
    $$

    where $h$ is the time step.
    $s$ can be interpreted as the spacing between two independent samples.

2. The exponential correlation time can be used to estimate the required length
   of the input sequences when computing an autocorrelation integral.
   The resolution of the frequency axis of the power spectrum is $1/T$,
   where $T=hN$ is the total simulation time,
   $h$ is the time step, and $N$ the number of steps.
   This resolution must be fine enough to resolve the zero-frequency peak
   associated with the exponential decay of the autocorrelation function.
   The width of the peak can be derived from [the Pade model](../theory/model.md)
   and is $1/2\pi\tau_\text{exp}$.
   To have ample frequency grid points in this first peak,
   the simulation time must be sufficiently long:

    $$
        T \gg 2\pi\tau_\text{exp}
    $$

    For example, $T = 20\pi\tau_\text{exp}$ will provide a decent resolution.

    Of course, before you start generating the data (e.g., through simulations),
    the value of $\tau_\text{exp}$ is yet unclear.
    Without prior knowledge of $\tau_\text{exp}$,
    you should first analyze preliminary data to get a first estimate of $\tau_\text{exp}$,
    after which you can plan the data generation more carefully.
    More details can be found in the section on [data sufficiency](../preparing_inputs/data_sufficiency.md).

    If you notice that your input sequences are many orders of magnitude longer than $\tau_\text{exp}$,
    the number of relevant frequency grid points in the spectrum can become impractically large.
    In this case, you can split up the input sequences into shorter parts with
    {py:func}`stacie.utils.split`.
    However, a better solution is to plan ahead more carefully
    and avoid sequences that are far longer than necessary.
    It is more efficient to generate more fully independent and shorter sequences instead.

    Note that $\tau_\text{exp}$ is also related to the block size
    when working with [block averages](../preparing_inputs/block_averages.md)
    to reduce storage requirements of production simulations.

## How to Compute with STACIE?

It is assumed that you can load one or (ideally) more
time-dependent sequences of equal length into a 2D NumPy array `sequences`.
Each row in this array is a sequence, and the columns correspond to time steps.
You also need to store the time step in a Python variable.
(If your data does not have a time step, just omit it from the code below.)

With these data, the autocorrelation times are computed as follows:

```python
import numpy as np
from stacie import compute_spectrum, estimate_acint, plot_results, PadeModel

# Load all the required inputs, the details of which will depend on your use case.
sequences = ...
timestep = ...

# Computation with STACIE.
spectrum = compute_spectrum(sequences, timestep=timestep)
result = estimate_acint(spectrum, PadeModel([0, 2], [2]))
print("Exponential autocorrelation time", result.corrtime_exp)
print("Uncertainty of the exponential autocorrelation time", result.corrtime_exp_std)
print("Integrated autocorrelation time", result.corrtime_int)
print("Uncertainty of the integrated autocorrelation time", result.corrtime_int_std)
```

A worked example can be found in the notebook
[Diffusion on a Surface with Newtonian Dynamics](../examples/surface_diffusion.py).
It also discusses the correlation times associated with the diffusive motion of the particles.
