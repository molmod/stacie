# Model Spectrum

STACIE supports three models for fitting the low-frequency part of the power spectrum.
In both models, the value at zero frequency corresponds to the autocorrelation integral.

1. The [ExpPolyModel](#stacie.model.ExpPolyModel) is the most general;
   it is an exponential function of a linear combination of simple monomials of the frequency.
   You can specify the degrees of the monomials, and typically a low degree works fine:

    - Degree $0$ is suitable for a white noise spectrum.
    - Degrees $\{0, 1\}$ can be used to extract useful information from a noisy spectrum.
    - Degrees $\{0, 1, 2\}$ are applicable to spectra with low statistical uncertainty,
      e.g., averaged over $>100$ inputs.
    - An even polynomial with degrees $\{0, 2\}$ is suitable for spectra
      that are expected to have a vanishing derivative at zero frequency.

    The main advantage of this model is its broad applicability,
    as it requires little prior knowledge of the functional form of the spectrum.

2. The [PadeModel](#stacie.model.PadeModel) is useful in several scenarios:

    - It can be configured to model a spectrum with a Lorentzian peak at the origin
      plus some white noise, which corresponds to an exponentially decaying {term}`ACF`.
      In this case, STACIE also derives the exponential correlation time,
      which can deviate from the integrated correlation time.

    - Rational functions are, in general, interesting because they can be
      parameterized to have well-behaved high-frequency tails,
      which can facilitate the regression.

3. The [LorentzModel](#stacie.model.LorentzModel) is a special case of the Pade model
   with a Lorentzian peak at the origin plus some white noise.
   It is equivalent to the Pade model with numerator degrees $\{0, 2\}$
   and denominator degrees $\{2\}$.
   This special case is not only implemented for convenience,
   since it is the most common way of using the Pade model,
   but also because it allows STACIE to derive the exponential correlation time.

(section-exppoly-target)=

## 1. ExpPoly Model

The [ExpPolyModel](#stacie.model.ExpPolyModel) is defined as:

$$
    I^\text{exppoly}_k = \exp\left(\sum_{s \in S} b_s f_k^s\right)
$$

where $S$ is the set of polynomial degrees, which must include 0.
With this form, $\exp(b_0)$ corresponds to the integral of the autocorrelation function.
When one obtains an estimate $\hat{b}_0$ and its variance $\hat{\sigma}^2_{b_0}$,
the autocorrelation integral is [log-normally distributed](https://en.wikipedia.org/wiki/Log-normal_distribution)
with estimated mean and variance:

$$
    \begin{aligned}
    \hat{\mathcal{I}}
    &= \exp\left(\hat{b}_0 + \frac{1}{2}\hat{\sigma}^2_{b_0}\right)
    \\
    \hat{\sigma}^2_{\mathcal{I}}
    &= \exp\left(2\hat{b}_0 + \hat{\sigma}^2_{b_0}\right)
        \left(\exp(\hat{\sigma}^2_{b_0}) - 1 \right)
    \end{aligned}
$$

To construct this model, you can create an instance of the `ExpPolyModel` class as follows:

```python
from stacie import ExpPolyModel
model = ExpPolyModel([0, 1, 2])
```

This model is identified as `exppoly(0, 1, 2)` in STACIE's screen output and plots.

(section-pade-target)=

## 2. Pade Model

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
and $S_\text{den}$ contains the polynomial degrees in the denominator, which must exclude 0.
With this model, $p_0$ corresponds to the integral of the autocorrelation function,
for which we simply have:

$$
    \begin{aligned}
    \hat{\mathcal{I}} &= \hat{p}_0
    \\
    \hat{\sigma}^2_{\mathcal{I}} &= \hat{\sigma}^2_{p_0}
    \end{aligned}
$$

To construct this model, you can create an instance of the `PadeModel` class as follows:

```python
from stacie import PadeModel
model = PadeModel([0, 2], [2])
```

This model is identified as `pade(0, 2; 2)` in STACIE's screen output and plots.

(section-lorentz-target)=

## 3. Lorentz Model

The [LorentzModel](#stacie.model.LorentzModel) is a special case of the Pade model
with numerator degrees $\{0, 2\}$ and denominator degrees $\{2\}$.
For this special case the model is equivalent to:

$$
    I^\text{lorentz}_k = A + \frac{B}{1 + (2 \pi f_k \tau_\text{exp})^2}
$$

where $f_k$ is the standard frequency grid of the discrete Fourier transform,
$A$ is the white noise level, $B$ is the amplitude of the Lorentzian peak,
and $\tau_\text{exp}$ is the exponential correlation time.
The frequency grid is defined as $f_k = k / (hN)$,
where $h$ is the time step of the discretized time axis, and $N$ is the number of samples.
We can write the Lorentzian model parameters in terms of the Pade model parameters as follows:

$$
    \begin{aligned}
        A &= \frac{\hat{p}_2}{\hat{q}_2}
        \\
        B &= \hat{p}_0 - \frac{\hat{p}_2}{\hat{q}_2}
        \\
        \tau_\text{exp} &= \frac{\sqrt{q_2}}{2 \pi}
    \end{aligned}
$$

The Pade model will only correspond to a Lorentzian peak if $q_2 > 0$ and $p_0 q_2 > p_2$.
When this is the case, $\tau_\text{exp}$ is related
to the width of the peak ($2 \pi \tau_\text{exp}$) in the power spectrum.
The exponential correlation time and its variance can then be derived
from the fitted parameters with first-order error propagation:

$$
    \begin{aligned}
    \hat{\tau}_\text{exp} &= \frac{\sqrt{\hat{q}_2}}{2 \pi}
    \\
    \hat{\sigma}^2_{\tau_\text{exp}} &= \frac{1}{16 \pi^2 \hat{q}_2} \hat{\sigma}^2_{q_2}
    \end{aligned}
$$

Note that this model is also applicable to data whose short-time correlations are not exponential,
as long as the tail of the ACF decays exponentially.
Such deviating short-time correlations will only affect the white noise level $A$
and features in the PSD at higher frequencies, which will be ignored by STACIE.

The implementation of the Lorentz model has the following advantages over the equivalent Pade model:

- The exponential correlation time and its uncertainty are computed.
- If no exponential correlation time can be computed,
  i.e. when $q_2 \le 0$ and $p_0 q_2 \le p_2$,
  the fit is not retained for the final average over all cutoff grid points.
- If the relative error of the exponential correlation time exceeds a certain threshold,
  which is set to $10 \%$ by default,
  the fit is not retained for the final average over all cutoff grid points.

The last two points weed out poor fits (typically at too low cutoff frequencies)
for which the maximum a posteriori (MAP) estimate
and the Laplace approximation of the posterior distribution tend to be unreliable.

To construct this model, you can create an instance of the `LorentzModel` class as follows:

```python
from stacie import LorentzModel
model = LorentzModel()
```

This model is identified as `lorentz(0.1)` in STACIE's screen output and plots,
where `0.1` is the relative error threshold for the exponential correlation time.
