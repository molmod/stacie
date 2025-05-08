# Model Spectrum

Stacie supports two models for fitting the low-frequency part of the power spectrum.
In both models, the value at zero frequency corresponds to the autocorrelation integral.

1. The [ExpPolyModel](#stacie.model.ExpPolyModel) is the most general:
   it is an exponential function of a linear combination of simple monomials of the frequency.
   One can specify the degrees of the monomials and typically a low degree works fine:

    - Degree `[0]` is suitable for a white noise spectrum.
    - Degrees `[0, 1]` can be used to extract useful information of a very noisy spectrum.
    - Degrees `[0, 1, 2]` is applicable to spectra with low statistical uncertainty,
      e.g. averaged over 100 inputs.
    - An even polynomial with degrees `[0, 2]` is suitable for spectra with a peak at zero frequency.

    The main advantage of this model is its broad applicability,
    as it requires little prior knowledge of the functional form of the spectra.

2. The Exponential Tail model is designed for autocorrelation functions that decay exponentially.
   Its primary advantage is that, in addition to the integrated correlation time,
   it also provides an estimate of the exponential correlation time.

## 1. ExpPolyModel

The [ExpPolyModel](#stacie.model.ExpPolyModel) is defined as

$$
    C^\text{exppoly}_k = \exp\left(\sum_{s\in S} a_s f_k^s\right)
$$

where $S$ is the set of polynomial degrees.
With this form, $a_0$ corresponds to the integral of the autocorrelation function.
When one obtains an estimate $\hat{a}_0$ and an estimated variance $\hat{C}(a_0)$,
the autocorrelation integral is [log-normally distributed](https://en.wikipedia.org/wiki/Log-normal_distribution)
with estimated mean and variance:

$$
    \hat{\mathcal{I}}
    &= \exp\left(\hat{a}_0 + \frac{1}{2}\hat{C}(a_0)\right)
    \\
    \hat{C}(\mathcal{I})
    &= \exp\left(2\hat{a}_0 + \hat{C}(a_0)\right)
    \left(\exp(\hat{C}(a_0)) - 1 \right)
$$

## 2. Exponential Tail Model

This model represents the autocorrelation function
as the sum of a short-term component and a (periodic) exponentially decaying tail:

$$
c_\Delta = c_\Delta^\text{short} + c_\Delta^\text{tail}
$$

### Short-term component

The short-term component is non-zero only for small positive or negative time lags:

$$
c_\Delta^\text{short} \neq 0 & \quad\text{if } |\Delta| \le \Delta_\text{short}
\\
c_\Delta^\text{short} = 0 & \quad\text{if } \Delta_\text{short} \lt |\Delta| \le N/2
$$

Because summations in DFT are commonly taken from $0$ to $N-1$, we rewrite this as:

$$
c_\Delta^\text{short}
    \neq 0 &\qquad \forall\,\Delta\in\lbrace
        0, \ldots, \Delta_\text{short},
        N-\Delta_\text{short}, \ldots, N-1
    \rbrace
\\
c_\Delta^\text{short}
    = 0 &\qquad \forall\,\Delta\in\lbrace
        \Delta_\text{short}+1, \ldots, N-\Delta_\text{short} -1
    \rbrace
$$

### Tail Component

The tail component is the periodic repetition (sum over $i$) of two exponential functions:
one decaying for positive time lags and one for negative time lags.
For $\Delta \in \lbrace 0, \ldots, N-1 \rbrace$, this can be written as:

$$
c_\Delta^\text{tail}
    = \frac{a_\text{tail}}{M} (r^\Delta + r^{N-\Delta}) \sum_{i=0}^\infty r^{iN}
$$

with $r < 1$.
The factor $\frac{1}{M}$ normalizes the tail component so that it sums to $a_\text{tail}$.

$$
M
    &= \left(\frac{1-r^N}{1-r} + r^N \frac{1-r^{-N}}{1-r^{-1}}\right) \sum_{i=0}^\infty r^{iN}
\\
    &= \left(1-r^{-N}\right) \frac{1 + r}{1-r} \sum_{i=0}^\infty r^{iN}
$$

We may absorb the infinite sum into the normalization constant to simplify the model:

$$
c_\Delta^\text{tail}
    & = \frac{a_\text{tail}}{M'} (r^\Delta + r^{N-\Delta})
\\
M'
    &= \left(1-r^{-N}\right) \frac{1 + r}{1-r}
$$

The exponential decay of the tail component is characterized by
its autocorrelation time, $\tau_\text{exp}$ {cite:p}`sokal_1997_monte`:

$$
    r = \exp\left(-\frac{h}{\tau_\text{exp}}\right)
$$

where $h$ is the time step.

### Discrete Fourier Transform

For small values of $k$, the discrete Fourier transform of the model autocorrelation function becomes:

$$
C^\text{exp-tail}_k
    &\approx
    a_\text{short} + \frac{a_\text{tail}}{M'} \left(
        \frac{1-r^N}{1 - r \omega^{-k}} + r^N\frac{1-r^{-N}}{1 - r^{-1} \omega^{-k}}
    \right)
\\
    &\approx
    a_\text{short} + \frac{a_\text{tail}}{M'} \left(
        \frac{1-r^N}{1 - r \omega^{-k}} + \frac{1-r^N}{1 - r \omega^k} - \left(1-r^N\right)
    \right)
\\
    &\approx
    a_\text{short} + \frac{a_\text{tail}}{M'} \left(1-r^N\right) \left(
         \frac{2 - r(\omega^k + \omega^{-k})}{1 - r(\omega^k + \omega^{-k}) + r^2} - 1
    \right)
\\
    &\approx
    a_\text{short} + a_\text{tail} \frac{1-r}{1+r} \left(
         \frac{2 - r(\omega^k + \omega^{-k})}{1 - r(\omega^k + \omega^{-k}) + r^2} - 1
    \right)
$$

For the first term we assumed $\omega^{k\Delta_\text{short}}\approx 1$.
Finally, we can substitute $\omega^k + \omega^{-k} = 2\cos(2\pi k/N)$:

$$
C^\text{exp-tail}_k
    &\approx
    a_\text{short} + a_\text{tail} \frac{1-r}{1+r} \left(
         2\frac{1 - r\cos(2\pi k/N)}{1 - 2r\cos(2\pi k/N) + r^2} - 1
    \right)
    \\
    &\approx
    a_\text{short} + a_\text{tail} \frac{(1-r)^2}{1 - 2r\cos(2\pi k/N) + r^2}
    \\
$$

This model can be fitted to the low-frequency part of an empirical autocorrelation function.
Once the parameters are fitted to an empirical spectrum, one finds:

$$
    \mathcal{I} \approx \hat{\mathcal{I}} = \hat{C}^\text{exp-tail}_0 = \hat{a}_\text{short} + \hat{a}_\text{tail}
$$

### Peak width

As illustrated in the
[Exponential Tail Model](../../examples/model.py)
example, the exponential decay of the autocorrelation function results in
a peak in the power spectrum at zero frequency.
The width of this peak at half the maximum of the tail term is found by solving:

$$
    1 - 2 r \cos(2\pi k_\text{half}/N) + r^2 = 2 (1 - r)^2
$$

The solution is:

$$
    k_\text{half}
        &= \frac{N}{2\pi}\arccos\left(\frac{4r - r^2 -1 }{2r}\right)
    \\
        &= \frac{N}{2\pi}\arccos\left(2 - \cosh\left(\frac{h}{\tau_\text{exp}}\right)\right)
$$

Using RFFT conventions, $f=k/hN$, this can be rewritten as a frequency:

$$
    f_\text{half}
        = \frac{1}{2\pi h}\arccos\left(2 - \cosh\left(\frac{h}{\tau_\text{exp}}\right)\right)
$$

We can rely on the following series expansion:

$$
    \arccos(2 - \cosh(u))
    \approx
    \left|u + \frac{1}{12} u^3 + \frac{1}{96} u^5 + \mathcal{O}(u^7)\right|
$$

Hence, in the limit $\tau_\text{exp} \gg h$, we find:

$$
    f_\text{half}
        \approx \frac{1}{2\pi \tau_\text{exp}}
$$

Sequences obtained from computer simulations
usually have sufficiently small time steps to satisfy this limit:
The steps must be able to resolve the fastest oscillations in the system
to correctly simulate its dynamics.
