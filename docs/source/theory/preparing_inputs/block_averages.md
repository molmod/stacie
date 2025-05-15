# Reducing Storage Requirements with Block Averages

When computer simulations generate time-dependent data,
they often use a time step that is much shorter than needed for an autocorrelation integral.
Storing (and processing) all the data may require too many resources.
To reduce the amount of data, we recommend taking block averages.
These block averages form a new time series with a time step equal to the block size
times the original time step.
This reduces the storage requirements by a factor equal to the block size.
If the program generating the sequences does not support block averages,
you can use {py:func}`stacie.utils.block_average`.

If the blocks are sufficiently small compared to the decay rate of the autocorrelation function,
STACIE will produce virtually the same results.
This should not be surprising,
since the closely related block averaging method {cite:p}`allen_2017_computer`
is commonly used for estimating the error on the mean of time-correlated data.

The effect of block averages can be understood by inserting them in the discrete power spectrum,
using STACIE's normalization convention to obtain the proper zero frequency limit.
Let $\hat{a}_m$ be the $m$'th block average of $L$ blocks with block size $B$.
For low frequencies (low indexes $k$), we make the following approximations:

$$
    \hat{I}_k
    &=
        F h \frac{1}{2} \sum_{\Delta=0}^{N-1} \hat{c}_\Delta \omega_N^{-kn}
    \\
    &=
        \frac{F h}{N} \frac{1}{2} \left|\sum_{n=0}^{N-1} \hat{x}_n \omega_N^{-kn}\right|^2
    \\
    &\approx
        \frac{F h}{N} \frac{1}{2} \left|\sum_{n=0}^{N-1} \hat{a}_{\lfloor n/B\rfloor} \omega_N^{-kn}\right|^2
    \\
    &\approx
        \frac{F h}{L} \frac{1}{2} \left| \sum_{\ell=0}^{L-1} B \hat{a}_\ell \omega_N^{-k \ell B}\right|^2
    \\
    &=
        \frac{F h B}{L} \frac{1}{2} \left| \sum_{\ell=0}^{L-1} \hat{a}_\ell \omega_L^{-k \ell}\right|^2
$$

with

$$
    \omega_N = \exp(i 2\pi/N) \qquad \omega_L = \exp(i 2\pi/L) = \omega_N^B
$$

The final result is the power spectrum of the block averages,
where $hB$ is the new time step and $L$ is the sequence length.

The approximations assume that $\omega_N^{kn}$ is nearly the same $\forall \, n \in [0, B]$.
Put differently, the approximation is small when

$$
    \omega_N^{kB} \approx 1
    \quad \text{or} \quad
    \frac{kB}{N} \ll 1
$$

Put differently, the larger the block size $B$,
the smaller the range of frequencies for which $\hat{I}_k$ is well approximated.

Depending on the model fitted to the spectrum,
there are two ways to determine the appropriate block size.

1. For any model, the number of points fitted to the spectrum
   is recommended to be about $20 \, P$,
   where $P$ is the number of parameters in the model.
   This means that the block size $B$ should be chosen such that

     $$
        B \ll \frac{N}{20 \, P}
     $$

     E.g. $B = \frac{N}{400 P}$ is a good choice.
     This practically means that there should be at least $400 \, P$ blocks.
     Fewer blocks will unavoidable lead to significant aliasing effects.

2. When using the [Pade model](../autocorrelation_integral/model.md#pademodel),
   one should ensure that the spectrum amplitudes $\hat{I}_k$ in the peak at zero frequency
   are not distorted by the block averages.
   The width of this peak in the Pade model is $1/2\pi\tau_\text{exp}$,
   and the resolution of the frequency axis of the power spectrum is $1/T$,
   where $T = hN$ is the total simulation time.
   These equations can be combined with $kB/N \ll 1$ to find:

    $$
         B \ll \frac{2\pi \tau_\text{exp}}{h}
    $$

    For example, $B = \frac{\pi \tau_\text{exp}}{10 h}$ will ensure that
    the relevant spectral features are preserved without any distortion
    in the spectrum derived from the block averages.

Just as with the required length of the input sequences,
a good choice of the block size cannot be determined *a priori*.
Also for the block size, a preliminary analysis with STACIE is recommended,
i.e. without any block averages initially.

An application of STACIE with block averages can be found in the following example notebook:
[Diffusion on a Surface with Newtonian Dynamics](../../examples/surface_diffusion.py).
