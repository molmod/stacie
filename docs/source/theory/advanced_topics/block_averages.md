# Reducing Storage Requirements with Block Averages

When computer simulations generate time-dependent data,
they often use a time step that is much shorter than needed for an autocorrelation integral.
Storing (and processing) all the data may require too much resources.
To reduce the amount of data, we recommend taking block averages.
These block averages form a new time series with a time step equal to the block size.
This reduces the storage requirements by a factor equal to the block size.

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
which is indeed true for slowly varying Fourier basis functions.
The last expression for the sample spectrum is simply that of the spectrum of the block averages,
with Stacie's normalization convention for a step size $hB$.

A good value for the block size is related to
[the exponential autocorrelation time](../properties/autocorrelation_time.md),
$\tau_\text{exp}$.
The block averages have a step size $Bh$,
which means that the highest frequency in the power spectrum of the block averages is $1/2 B h$,
due to the [Nyquist-Shannon Sampling Theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem).
(This is also the maximum RFFT frequency.)
This maximum frequency should be high enough to fully contain
the [peak at zero frequency](../autocorrelation_integral/model.md#peak-width)
associated with the exponential decay of the autocorrelation function.
Thus, we find $1/2 B h \gg 1/2\pi \tau_\text{exp}$ or:

$$
    B \ll \frac{\pi \tau_\text{exp}}{h}
$$

For example, $B = \frac{1}{20}\times\frac{\pi \tau_\text{exp}}{h}$ will ensure that
all the relevant features are present without any distortion
in the spectrum derived from the block averages.
Hence, when estimating $\tau_\text{exp}$ for a
[a suitable simulation length](../properties/autocorrelation_time.md),
one can also fix a suitable block size.

Larger block sizes will generally lead to worse results:

- The approximations in the derivation above will become worse.
  This will distort the low-frequency spectrum,
  so that the exponential tail model can only be fitted to a smaller portion.

- Useful information is lost in the block averages.
  The statistical uncertainties can be reduced by decreasing the block size.

- For very large blocks, one essentially obtains a white noise spectrum.
  Fitting an Exponential Tail Model to white noise can lead to overfitting.
  If you have no choice, you can circumvent this issue by specifying a
  {py:class}`stacie.model.WhiteNoiseModel` instance
  as an optional argument to the {py:func}`stacie.estimate.estimate_acint` function.
  This workaround is not ideal because
  the (slowest) relaxation time can no longer be derived from such block averages.

An application of Stacie with block averages can be found in the following example notebook:
[Diffusion on a Surface with Newtonian Dynamics](../../examples/surface_diffusion.py).
