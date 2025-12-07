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

2. The [PadeModel](#stacie.model.PadeModel) approximates the spectrum as a rational function,
   i.e., the ratio of two polynomials.
   Rational functions can be parameterized to have well-behaved high-frequency tails,
   which can facilitate the regression.

3. The [LorentzModel](#stacie.model.LorentzModel) is a special case of the Padé model
   with a Lorentzian peak at the origin plus some white noise.
   It is equivalent to the Padé model with numerator degrees $\{0, 2\}$
   and denominator degrees $\{2\}$.
   This special case is not only implemented for convenience,
   since it is the most common way of using the Padé model,
   but also because it allows STACIE to derive the exponential correlation time.

(section-exppoly-target)=

## 1. ExpPoly Model

The [ExpPolyModel](#stacie.model.ExpPolyModel) is defined as:

$$
    I^\text{exppoly}(f) = \exp\left(\sum_{s \in S} b_s f^s\right)
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
    I^\text{pade}(f) = \frac{
        \displaystyle
        \sum_{s \in S_\text{num}} p_s f^s
    }{
        \displaystyle
        1 + \sum_{s \in S_\text{den}} q_s f^s
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

The [LorentzModel](#stacie.model.LorentzModel) assumes that for large time lags,
$\Delta_t$, the autocorrelation function (ACF) exhibits an exponential decay
superimposed with a white noise background:

$$
  C(\Delta_t) = C_0\, \delta(\Delta_t) + C_1 \exp\left(-\frac{|\Delta_t|}{\tau_\text{exp}}\right)
$$

where $C_0$ is the white noise level, $C_1$ is the amplitude of the exponential decay,
and $\tau_\text{exp}$ is the exponential correlation time.
The Lorentz model is essentially the corresponding power spectral density:

$$
    I^\text{lorentz}(f) = C_0 + \frac{2 C_1 \tau_\text{exp}}{1 + (2 \pi f \tau_\text{exp})^2}
$$

This model can be expressed as a special case of the Padé model,
with numerator degrees $\{0, 2\}$ and denominator degrees $\{2\}$, and is fitted accordingly.
The Padé model will only correspond to a Lorentzian peak if $q_2 > 0$ and $p_0 q_2 > p_2$.
When this is the case, $\tau_\text{exp}$ is related
to the width of the peak ($2 \pi \tau_\text{exp}$) in the power spectrum.
When these conditions are met after the regression,
the parameters of the Padé model are converted as follows:

$$
    \begin{aligned}
        \hat{C}_0 &= \frac{\hat{p}_2}{\hat{q}_2}
        \\
        \hat{C}_1 &= \frac{\pi}{\sqrt{\hat{q}_2}}\left(\hat{p}_0 - \frac{\hat{p}_2}{\hat{q}_2}\right)
        \\
        \tau_\text{exp} &= \frac{\sqrt{q_2}}{2 \pi}
    \end{aligned}
$$

The covariance of these parameters is derived from the covariance of the Padé parameters
using first-order error propagation.
We first introduce the following Jacobian matrix:

$$
    \begin{aligned}
        J &=
        \begin{pmatrix}
            \frac{\partial C_0}{\partial p_0} &
            \frac{\partial C_0}{\partial p_2} &
            \frac{\partial C_0}{\partial q_2} \\
            \frac{\partial C_1}{\partial p_0} &
            \frac{\partial C_1}{\partial p_2} &
            \frac{\partial C_1}{\partial q_2} \\
            \frac{\partial \tau_\text{exp}}{\partial p_0} &
            \frac{\partial \tau_\text{exp}}{\partial p_2} &
            \frac{\partial \tau_\text{exp}}{\partial q_2}
        \end{pmatrix}
        =
        \begin{pmatrix}
            0 &
            \frac{1}{q_2} &
            -\frac{p_2}{q_2^2}
            \\
            \frac{\pi}{\sqrt{q_2}} &
            -\frac{\pi}{q_2^{3/2}} &
            \frac{\pi}{2 q_2^{3/2}}\left(\frac{3p_2}{q_2} - p_0\right)
            \\
            0 &
            0 &
            \frac{1}{4 \pi \sqrt{q_2}}
        \end{pmatrix}
    \end{aligned}
$$

and use it to compute the covariance of the Lorentz parameters:

$$
    \hat{\mathbf{C}}_\text{lorentz} =
    J \,
    \hat{\mathbf{C}}_\text{pade(0, 2; 2)} \,
    J^T
$$

Note that this model is also applicable to data whose short-time correlations are not exponential,
as long as the tail of the ACF decays exponentially.
Such deviating short-time correlations will only affect the white noise level $\hat{C}_0$
and features in the PSD at higher frequencies, which are ignored by STACIE.

The implementation of the Lorentz model has the following advantages over the equivalent Padé model:

- The exponential correlation time and its uncertainty are computed.
- If no exponential correlation time can be computed,
  i.e. when $\hat{q}_2 \le 0$ and $\hat{p}_0\, \hat{q}_2 \le \hat{p}_2$,
  the fit is not retained for the final average over all cutoff grid points.
- After the optimization of the model parameters at a given frequency cutoff,
  the following two heuristics are applied to exclude or downweight fits
  with a high relative error of the exponential correlation time:

    - If the relative error of the exponential correlation time is 100 times larger
      than that of the autocorrelation integral, the fit is discarded.
    - In all other cases, a penalty is added to the cutoff criterion,
      which is proportional to the ratio of the relative error of the exponential correlation time
      and the autocorrelation integral.

  This heuristic is based on the observation that large relative errors of
  the exponential correlation time often indicate poor fits for which the MAP estimate
  and the Laplace approximation of the posterior distribution tend to be unreliable.
  By taking a ratio of relative errors, the penalty is dimensionless
  and insensitive to the overall uncertainty of the spectrum.

  The result of these two heuristics is that the final estimate of the exponential correlation time,
  after averaging over all frequency cutoffs, is more robust and reliable.

  Note that the hyperparameters of the penalty can be altered, but it is recommended to keep
  their default values.

To construct this model, you can create an instance of the `LorentzModel` class as follows:

```python
from stacie import LorentzModel
model = LorentzModel()
```

This model is identified as `lorentz()` in STACIE's screen output and plots.
