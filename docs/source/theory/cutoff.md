# Frequency Cutoff

In STACIE, a model is fitted to the low-frequency part of the sampling {term}`PSD`.
This low-frequency part is defined by a cutoff frequency, $f_{\text{cut}}$,
above which the model is not expected to explain the data.
The [previous section](statistics.md) discussed how to implement a local regression
using a smooth switching function parameterized by such a cutoff frequency, $f_\text{cut}$.
A good choice for the cutoff seeks a trade-off between two conflicting goals:

1. If too much data is included in the fit,
   the model may be too simple to explain all features of the spectrum.
   It underfits the data, and the estimates are generally biased.
2. If too little data is included in the fit,
   the variance of the estimated parameters is larger than necessary,
   meaning that not all relevant information is used.

Finding a good compromise between these two can be done in several ways,
and similar difficulties arise in other approaches to compute transport coefficients.
For example, in the direct quadrature of the {term}`ACF`,
the truncation of the integral faces a similar trade-off.

Because the [model](model.md) is fitted to a sampling PSD with known and convenient statistical properties,
as discussed in the [previous section](statistics.md),
it is possible to determine the cutoff frequency systematically.
As also explained in the [previous section](statistics.md),
the cutoff frequency is not a proper hyperparameter in the Bayesian sense,
meaning that a straightforward marginalization over the cutoff frequency is not possible
{cite}`rasmussen_2005_gaussian`.
Instead, STACIE uses cross-validation to find a good compromise between bias and variance.
As explained below, a model likelihood is constructed based on cross-validation,
whose unit is independent of the cutoff frequency.
This model is then used to marginalize estimated parameters over the cutoff frequency.

## Effective number of fitting points

The concept of "effective number of fitting points" is used regularly in the following subsections.
For a given cutoff frequency, it is defined as:

$$
    N_{\text{eff}}(f_{\text{cut}}) = \sum_{k=1}^{M} w(f_k|f_{\text{cut}})
$$

This is simply the sum of the weights introduced in the section on [regression](statistics.md#regression).

## Grid of Cutoff Frequencies

STACIE uses a logarithmic grid of cutoff frequencies and fits model parameters for each cutoff.
The grid is defined as:

$$
    f_{\text{cut},j} = f_{\text{cut},0} \, r^j
$$

where $f_{\text{cut},0}$ is the lowest cutoff frequency in the grid,
and $r$ is the ratio between two consecutive cutoff frequencies.
The following parameters define the grid:

- The lowest cutoff is determined by solving:

    $$
        N_{\text{eff}}(f_{\text{cut,min}}) = N_{\text{eff, min}}
    $$

    where $N_{\text{eff, min}}$ is a user-defined parameter, and $P$ is the number of model parameters.
    In STACIE, the default value is $N_{\text{eff, min}} = 5P$,
    which reduces the risk of numerical issues in the regression.
    The value of $N_{\text{eff, min}}$ can be adjusted using the `neff_min` option
    in the function [`estimate_acint()`](#stacie.estimate.estimate_acint).

- The maximum cutoff frequency is determined by solving:

    $$
        N_{\text{eff}}(f_{\text{cut,max}}) = N_{\text{eff, max}}
    $$

    where $N_{\text{eff, min}}$ is a user-defined parameter.
    In STACIE, the default value is $N_{\text{eff, min}} = 1000$.
    This value can be modified using the `neff_max` option
    in the function [`estimate_acint()`](#stacie.estimate.estimate_acint).
    The purpose of this parameter is to limit the computational cost of the regression.
    (For short inputs, the highest cutoff frequency is also constrained by the Nyquist frequency.)

- The ratio between two consecutive cutoff frequencies is:

    $$
        r = \exp(g_\text{sp}/\beta)
    $$

    where $g_\text{sp}$ is a user-defined parameter,
    and $\beta$ controls the steepness of the switching function $w(f|f_{\text{cut}})$.
    In STACIE, the default value is $g_\text{sp} = 0.5$.
    This value can be adjusted using the `fcut_spacing` option in the function [`estimate_acint()`](#stacie.estimate.estimate_acint).
    By incorporating the parameter $\beta$ into the definition of $r$,
    a steeper switching function automatically requires a finer grid of cutoff frequencies.

Parameters are fitted for all cutoffs, starting from the lowest one.
As shown below, the scan of the cutoff frequencies can be stopped before reaching $f_{\text{cut,max}}$.

## Cross-Validation

Given a cutoff frequency, $f_{\text{cut},j}$,
STACIE estimates model parameters $\hat{\mathbf{b}}^{(j)}$
and their covariance matrix $\hat{\mathbf{C}}_{\mathbf{b}^{(j)},\mathbf{b}^{(j)}}$.
To quantify the degree of over- or underfitting,
the model parameters are further refined
by fitting them to the first and second halves of the low-frequency part of the sampling PSD.
To make these refinements robust, the two halves are defined using smooth switching functions:

$$
    w_{\text{left}}(f|f_{\text{cut},j}) &= w(f|g_\text{cv} f_{\text{cut},j} / 2)
    \\
    w_{\text{right}}(f|f_{\text{cut},j}) &= w(f|g_\text{cv} f_{\text{cut},j}) - w_{\text{left}}(f|f_{\text{cut},j})
$$

The parameter $g_\text{cv}$ is a user-defined parameter
that controls the amount of data used in the refinements.
In STACIE, the default value is $g_\text{cv} = 1.25$,
meaning that 25% more data is used compared to the original fit.
(This makes the cross-validation more sensitive to underfitting,
which has been found beneficial in practice.)
This parameter can be controlled using the `fcut_factor` option
in the [`CV2LCriterion`](#stacie.cutoff.CV2LCriterion) class.
An instance of this class can be passed to the `cutoff_criterion` argument
of the function [`estimate_acint()`](#stacie.estimate.estimate_acint).

Instead of performing two full non-linear regressions of the parameters for the two halves,
linear regression is used to make first-order approximations of the changes in parameters.
For cutoffs leading to well-behaved fits, these corrections are small,
justifying the use of a linear approximation.

The design matrix of the linear regression is:

$$
    D_{kp} = \left.
            \frac{
                \partial I^\text{model}(f_k; \mathbf{b})
            }{
                \partial b_p
            }
        \right|_{\mathbf{b} = \hat{\mathbf{b}}^{(j)}}
$$

The expected values are the residuals between the sampling PSD and the model:

$$
    y_k = \hat{I}_k - I^\text{model}(f_k; \hat{\mathbf{b}}^{(j)})
$$

The measurement error is the standard deviation of the Gamma distribution,
using the model spectrum in the scale parameter and the shape parameter of the sampling PSD:

$$
    \sigma_k = \frac{I^\text{model}(f_k; \hat{\mathbf{b}}^{(j)})}{\sqrt{\alpha_k}}
$$

The weighted regression to obtain first-order corrections to the parameters $\hat{\mathbf{b}}^{(j)}$
solves the following linear system in the least-squares sense:

$$
    \frac{w_k}{\sigma_k} \sum_{p=1}^{P} D_{kp}\, \hat{b}^{(j)}_{\text{corr},p}
    = \frac{w_k}{\sigma_k} y_k
$$

where $w_k$ is the weight of the $k$-th frequency point.
This system is solved once with weights for the left half and once for the right half.

The function [`linear_weighted_regression()`](#stacie.cutoff.linear_weighted_regression)
provides a robust pre-conditioned implementation of the above linear regression.
It can handle multiple weight vectors simultaneously
and can directly compute linear combinations of parameters for different weight vectors.
It is used to directly compute the difference between the corrections for the left and right halves,
denoted as $\hat{\mathbf{d}}$, and its covariance matrix $\hat{\mathbf{C}}_{\mathbf{d},\mathbf{d}}$.
Normally, the model parameters fitted to both halves must be the same,
and the negative log-likelihood of the fitted parameters being identical is given by:

$$
    \operatorname{criterion}^\text{CV2L} = -\ln \mathcal{L}^\text{CV2L}\left(
        \hat{\mathbf{d}}^{(j)},
        \hat{\mathbf{C}}^{(j)}_{\mathbf{d}}
    \right)
    = \frac{P}{2}\ln(2\pi)
      +\underbrace{\frac{1}{2}\ln\left|\hat{\mathbf{C}}^{(j)}_{\mathbf{d}}\right|}_\text{variance}
      +\underbrace{
        \frac{1}{2}
        \bigl(\hat{\mathbf{d}}^{(j)}\bigr)^T
        \bigl(\hat{\mathbf{C}}^{(j)}_{\mathbf{d}}\bigr)^{-1}
        \hat{\mathbf{d}}^{(j)}
      }_\text{bias}
$$

When starting from the lowest cutoff grid point,
the second term of the criterion (the variance term) will be high
because the parameters are poorly constrained by the small amount of data used in the fit.
As the cutoff frequency and the effective number of fitting points increase,
the model becomes better constrained.
The second term will decrease, but as soon as the model underfits the data,
the third term (the bias term) will steeply increase.
Practically, the cutoff scan is interrupted
when the criterion exceeds the incumbent by $g_\text{incr}$.
The default value is $g_\text{incr} = 100$,
but this can be changed using the `criterion_high` option
in the function [`estimate_acint()`](#stacie.estimate.estimate_acint).

A good cutoff frequency is the one that minimizes the criterion,
thereby finding a good compromise between bias and variance.

```{note}
In the description above, we assume that a cutoff exists
for which the model can explain the spectrum.
With unfortunate model choices, this may not be the case.
The cutoff scan will then try to find a compromise between the bias and variance,
but this will not be useful if the model can not be describe the spectrum at all.
This situation can be detected by checking the Regression Cost Z-score,
derived in the previous section.
```

## Marginalization Over the Cutoff Frequency

Any method to deduce the cutoff frequency from the spectrum,
whether it is human judgment or an automated algorithm,
introduces some {term}`uncertainty` in the final result
because the cutoff is based on a sampling PSD spectrum with statistical uncertainty.

In STACIE, this uncertainty is accounted for
by marginalizing the model parameters over the cutoff frequency,
using $\mathcal{L}^\text{CV2L}$ as a model for the likelihood.
This approach naturally incorporates the uncertainty in the cutoff frequency
and is preferred over fixing the cutoff frequency at a single value.

Practically, the final estimate of the parameters and their covariance is computed
using [standard expressions for mixture distributions](https://en.wikipedia.org/wiki/Mixture_distribution#Moments):

$$
  \begin{split}
    \hat{\mathbf{b}} &= \sum_{j=1}^J W_j\, \hat{\mathbf{b}}^{(j)}
    \\
    \hat{C}_{\mathbf{b},\mathbf{b}} &= \sum_{j=1}^J W_j\, \left(
      \hat{C}_{\mathbf{b}^{(j)},\mathbf{b}^{(j)}}
      + (\hat{\mathbf{b}} - \hat{\mathbf{b}}^{(j)})^2
    \right)
  \end{split}
$$

Here, $\hat{\mathbf{b}}^{(j)}$ and $\hat{C}_{\mathbf{b}^{(j)},\mathbf{b}^{(j)}}$ represent
the parameters and their covariance, respectively, for cutoff $j$.
The weights $W_j$ sum to 1 and are proportional to $\mathcal{L}^\text{CV2L}$.

Note that STACIE also computes weighted averages of other quantities in the same way,
including:

- The effective number of fitting points, $N_\text{eff}$
- The cutoff frequency, $f_\text{cut}$
- The switching function, $w(f|f_\text{cut})$
- The regression cost Z-score, $Z_\text{cost}$
- The cutoff criterion Z-score, $Z_\text{criterion}$ (defined below)

## Cutoff Criterion Z-score

In the far majority of cases, the cutoff criterion will be dominated by the bias term:
it typically increases steeply as soon as the model underfits the data.
In contrast, the variance term decreases relatively slowly.
As a result, the minimum of the cutoff criterion is well-defined at the onset of underfitting.
This works particularly well when the spectrum can be computed with a high frequency resolution.
Unfortunately, there are cases where this ideal pattern does not hold,
usually when providing too little data to STACIE,
in which the cutoff criterion exhibits statistical fluctuations.
To detect such ill-constrained cases, STACIE computes a Z-score for the cutoff criterion.
This Z-score is defined as in the same spirit as the Regression Cost Z-score,
but now using $\operatorname{criterion}^\text{CV2L}$ instead of the regression cost function.

The cutoff criterion Z-score is defined as:

$$
    Z_\text{criterion} = \frac{
        \operatorname{criterion}^\text{CV2L} - \mean\left[
            \hat{\operatorname{criterion}}^\text{CV2L}
        \right]
    }{
        \std\left[
            \hat{\operatorname{criterion}}^\text{CV2L}
        \right]
    }
$$

The mean and standard deviation are computed by averaging over all vectors $\mathbf{d}$
from the likelihood $\mathcal{L}^\text{CV2L}$.

For cutoff frequencies that minimize the criterion, the Z-score should be close to zero,
because the difference between two parameters fitted to the left and right halves of the spectrum
should be zero within the statistical uncertainty.
When the bias term in the cutoff criterion is noisy,
it may feature minima dominated by the variance term,
which are not useful and may produce unreliable estimates (with misleading error bars).
In such cases, the cutoff criterion Z-score will be significantly larger than zero.

The mean in the Z-score can be worked out easily
because it corresponds to the entropy of a multivariate normal distribution:

$$
    \mean\left[
        \hat{\operatorname{criterion}}^\text{CV2L}
    \right]
    = \mean_{\mathbf{d}}\left[
        -\ln \mathcal{L}^\text{CV2L}\left(
            \hat{\mathbf{d}},
            \hat{\mathbf{C}}_{\mathbf{d},\mathbf{d}}
        \right)
    \right]
    = \frac{P}{2}\ln(2\pi e) + \frac{1}{2}\ln|\hat{\mathbf{C}}_{\mathbf{d},\mathbf{d}}|
$$

For the standard deviation, we first work out the variance of the cutoff criterion:

$$
    \var\left[
        \hat{\operatorname{criterion}}^\text{CV2L}
    \right]
    = \var_{\mathbf{d}} \left[
        -\ln \mathcal{L}^\text{CV2L}\left(
            \hat{\mathbf{d}},
            \hat{\mathbf{C}}_{\mathbf{d},\mathbf{d}}
        \right)
    \right]
$$

Only the bias term contributes to the variance of the cutoff criterion.
This term can be rewritten as one half the sum of $P$ squared standard normal distributed variables.
By making use of the properties of the chi-squared distribution,
we can work out the variance of the bias term
and take the square root to obtain the standard deviation:

$$
    \std\left[
        \hat{\operatorname{criterion}}^\text{CV2L}
    \right]
    = \sqrt{\frac{P}{2}}
$$
