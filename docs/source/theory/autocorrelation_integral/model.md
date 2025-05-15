# Model Spectrum

STACIE supports two models for fitting the low-frequency part of the power spectrum.
In both models, the value at zero frequency corresponds to the autocorrelation integral.

1. The [ExpPolyModel](#stacie.model.ExpPolyModel) is the most general:
   it is an exponential function of a linear combination of simple monomials of the frequency.
   One can specify the degrees of the monomials, and typically a low degree works fine:

    - Degree `[0]` is suitable for a white noise spectrum.
    - Degrees `[0, 1]` can be used to extract useful information from a very noisy spectrum.
    - Degrees `[0, 1, 2]` are applicable to spectra with low statistical uncertainty,
      e.g., averaged over 100 inputs.
    - An even polynomial with degrees `[0, 2]` is for spectra
      that are expected to have a vanishing derivative at zero frequency.

    The main advantage of this model is its broad applicability,
    as it requires little prior knowledge of the functional form of the spectrum.

2. The [PadeModel](#stacie.model.PadeModel) is useful in several scenarios:

    - It can be configured to model a spectrum with a Lorentzian peak at the origin
      plus some white noise, which corresponds to an exponentially decaying ACF.
      In this case, STACIE can derive the exponential correlation time.

    - Rational functions are, in general, interesting because they can be
      parameterized to have well-behaved high-frequency tails,
      which can facilitate the regression.

## 1. ExpPolyModel

The [ExpPolyModel](#stacie.model.ExpPolyModel) is defined as:

$$
    I^\text{exppoly}_k = \exp\left(\sum_{s \in S} b_s f_k^s\right)
$$

where $S$ is the set of polynomial degrees, which must include 0.
With this form, $b_0$ corresponds to the integral of the autocorrelation function.
When one obtains an estimate $\hat{b}_0$ and its variance $\hat{\sigma}^2_{b_0}$,
the autocorrelation integral is [log-normally distributed](https://en.wikipedia.org/wiki/Log-normal_distribution)
with estimated mean and variance:

$$
    \hat{\mathcal{I}}
    &= \exp\left(\hat{b}_0 + \frac{1}{2}\hat{\sigma}^2_{b_0}\right)
    \\
    \hat{\sigma}^2_{\mathcal{I}}
    &= \exp\left(2\hat{b}_0 + \hat{\sigma}^2_{b_0}\right)
        \left(\exp(\hat{\sigma}^2_{b_0}) - 1 \right)
$$

To construct this model, you can create an instance of the `ExpPolyModel` class as follows:

```python
from stacie.model import ExpPolyModel
model = ExpPolyModel(degrees=[0, 1, 2])
```

## 2. PadeModel

The [PadeModel](#stacie.model.PadeModel) is defined as:

$$
    I^\text{pade}_k = \frac{
        \displaystyle
        \sum_{s \in S_\text{num}} p_s f_k^s
    }{
        \displaystyle
        1 + \sum_{s \in S_\text{den}} q_s f_k^s
    }
$$

where $S_\text{num}$ contains the polynomial degrees in the numerator, which must include 0,
and $S_\text{den}$ contains the polynomial degrees in the denominator, which must not include 0.
With this model, $p_0$ corresponds to the integral of the autocorrelation function,
for which we simply have:

$$
    \hat{\mathcal{I}} &= \hat{p}_0
    \\
    \hat{\sigma}^2_{\mathcal{I}} &= \hat{\sigma}^2_{p_0}
$$

For the special case of a Lorentzian peak at the origin plus some white noise,
that is $S_\text{num} = \{0, 2\}$ and $S_\text{den} = \{2\}$,
the model is equivalent to:

$$
    I^\text{lorentz}_k = A + \frac{B}{1 + (2 \pi f_k \tau_\text{exp})^2}
$$

where $f_k$ is the standard frequency grid of the discrete Fourier transform,
$f_k = k / (hN)$, $h$ is the time step of the discretized time axis,
and $N$ is the number of samples.
The parameter $A$ is the white noise level, and $B$ is the amplitude of the Lorentzian peak.
Of particular interest is the correlation time $\tau_\text{exp}$.
When fitting the PadeModel with $S_\text{num} = \{0, 2\}$ and $S_\text{den} = \{2\}$,
the exponential correlation time and its variance can be derived
from the fitted parameters as with first-order error propagation:

$$
    \tau_\text{exp} &= \frac{\sqrt{\hat{q}_2}}{2 \pi}
    \\
    \hat{\sigma}^2_{\tau_\text{exp}} &= \frac{1}{16 \pi^2 \hat{q}_2} \hat{\sigma}^2_{q_2}
$$
