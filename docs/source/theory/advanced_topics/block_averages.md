# Reducing Storage Requirements with Block Averages

When computer simulations generate time-dependent data,
they often use a time step that is much shorter than needed for an autocorrelation integral.
Storing (and processing) all data may require too many resources.
To reduce the amount of data, we recommend taking block averages.
These block averages form a new time series with a time step corresponding to the block size.
If the blocks are sufficiently small compared to the decay rate of the autocorrelation function,
Stacie will produce virtually the same results for all parameters of the Exponential Tail Model.
This should not be surprising,
since the block averaging method is commonly used for practically the same purpose as Stacie.

The effect of block averages can be understood by inserting them in the discrete power spectrum,
using Stacie's normalization convention to obtain the proper zero frequency limit.
Let $\hat{y}_m$ be the $m$'th block average of $M$ blocks with block size $B$.
For low frequencies (low indexes $k$), we make the following approximations:

$$
    \hat{C}_k
    &=
        F h \sum_{\Delta=0}^{N-1} \hat{c}_\Delta \omega^{-kn}
    \\
    &=
        \frac{F h}{N}\left|\sum_{n=0}^{N-1} \hat{x}_n \omega_N^{-kn}\right|^2
    \\
    &\approx
        \frac{F h}{N} \left|\sum_{n=0}^{N-1} \hat{y}_{\lfloor n/B\rfloor} \omega_N^{-kn}\right|^2
    \\
    &\approx
        \frac{F h}{N} \left| \sum_{m=0}^{M-1} B \hat{y}_m \omega_N^{-kmB}\right|^2
    \\
    &\approx
        \frac{F h B}{M} \left| \sum_{m=0}^{M-1} \hat{y}_m \omega_M^{-km}\right|^2
$$

with

$$
    \omega_N = \exp(i 2\pi/N) \qquad \omega_M = \exp(i 2\pi/M)
$$

The approximations assume that for small indexes $k$,
$\omega_N^{-kn}$ is nearly independent of $n$,
which is only true for slowly varying Fourier basis functions.
The last expression for the sample spectrum is simply that of the spectrum of the block averages,
with Stacie's normalization convention for a step size $hB$.

Increasing the block size even further is possible, but less useful.
This will gradually lead to block averages that resemble white noise as the block size increases.
The (slowest) relaxation time can no longer be derived from such block averages.
The integral of the autocorrelation function can still be estimated correctly,
but the statistical uncertainty will increase with the block size.
Therefore, it is recommended to keep the block size well below
the decay time of the exponential tail model (e.g., factor 20).

An application of Stacie with block averages can be found in the following example notebook:
[Diffusion on a Surface with Newtonian Dynamics](../../examples/surface_diffusion.py).
