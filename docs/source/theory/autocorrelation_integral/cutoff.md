# Frequency Cutoff

In STACIE, a model must be fitted to the low-frequency part of the sampling PSD.
The [previous section](statistics.md) discussed how to implement a local regression
using a smooth switching function parametrized by a cutoff frequency, $f_\text{cut}$.
A good choice of the cutoff seeks a trade-off between two conflicting goals:

1. When too much data is included in the fit,
   the model is too simple to explain all the features of the spectrum.
   It underfits the data and the estimates are in general biased.
2. When too little data is included in the fit,
   the variance of the estimated parameters is larger than needed,
   meaning that not all relevant information is used.

Finding a good compromise between the two can be done in several ways,
and similar difficulties can be found in other approaches to compute transport coefficients.
For example, in the direct quadrature of the ACF, the truncation of the integral faces the same problem.

Because the [model](model.md) is fitted to a sampling PSD with known and convenient statistical properties,
as discussed in the [previous section](statistics.md),
it becomes possible to determine the cutoff frequency systematically.
As also explained in the [previous section](statistics.md),
the cutoff frequency is not a proper hyperparameter in the Bayesian sense,
meaning that a straightforward of marginalization over the cutoff frequency is not possible
{cite}`rasmussen_2005_gaussian`.
Instead, we propose to use cross validation to find a good compromise between bias and variance.
As explained below, we use cross-validation to construct a model likelihood,
whose unit is independent of the cutoff frequency,
and which can be used to marginalize estimated parameters over the cutoff frequency.

## Effective number of fitting points

The following subsections regularly use the concept "effective number of fitting points".
We define it for a given cutoff frequency, as:

$$
    N_{\text{eff}}(f_{\text{cut}}) = \sum_{k=1}^{M} w(f_k|f_{\text{cut}})
$$

This is simply the sum of the weights introduced in
the [regression](statistics.md#regression) of the model to the sampling PSD.

## Grid of cutoff frequencies

STACIE fits models for a logarithmic grid of cutoff frequencies, defined as:

$$
    f_{\text{cut},j} = f_{\text{cut},0} \, r^j
$$

where $f_{\text{cut},0}$ is the lowest cutoff frequency in the grid
and $r$ is the ratio between two consecutive cutoff frequencies.
The following parameters control the parameters of the grid:

- The lowest cutoff considered is fixed by solving

    $$
        N_{\text{eff}}(f_{\text{cut,min}}) = g_\text{min} P
    $$

    where $g_\text{min}$ is a user-defined parameter and $P$ is the number model parameters.
    In STACIE, the default value is $g_\text{min} = 5$
    to reduce the risk of numerical issues in the regression.
    One can change the value of $g_\text{min} P$ with the option `neff_min`
    of the function [`estimate_acint()`](#stacie.estimate.estimate_acint).

- The maximum cutoff frequency is fixed by solving

    $$
        N_{\text{eff}}(f_{\text{cut,max}}) = g_\text{max}
    $$

    where $g_\text{max}$ is a user-defined parameter.
    In STACIE, the default value is $g_\text{max} = 1000$.
    One can change the value of $g_\text{max}$ with the option `neff_max`
    of the function [`estimate_acint()`](#stacie.estimate.estimate_acint).
    The sole purpose of this parameter is to control the computional cost of the regression.
    (For short inputs, the highest cutoff frequency is also limited by the Nyquist frequency.)

- The ratio between two consecutive cutoff frequencies is:

    $$
        r = \exp(g_\text{sp}/\beta)
    $$

    where $g_\text{sp}$ is a user-defined parameter
    and $\beta$ controls the steepness of the switching function $w(f|f_{\text{cut}})$.
    In STACIE, the default value is $g_\text{sp} = 0.5$.
    One can change the value of $g_\text{sp}$ with the option `fcut_spacing`
    of the function [`estimate_acint()`](#stacie.estimate.estimate_acint).
    By incorporating the parameter $\beta$ in the definition of $r$,
    we automatically ensure that a steeper switching function will require a finer grid of cutoff frequencies.

Parameters are fitted for all cutoffs, starting for the lowest one.
As shown, below one can terminate the scan of the cutoff frequencies
well before reaching the maximum cutoff frequency.

## Cross-validation

Given a cutoff frequency, $f_{\text{cut},j}$, STACIE estimates model parameters
$\hat{\mathbf{b}}^{(j)}$ and their covariance matrix $\hat{\mathbf{C}}_{\mathbf{b}^{(j)},\mathbf{b}^{(j)}}$.
To quantify the degree of over- or underfitting, the model parmaeters are further refined
by refitting them to the first and the second halves of the low-frequency part of the sampling PSD.
To make these refinements robust, the two halves are also defined through smooth switching functions:

$$
    w_{\text{left}}(f|f_{\text{cut},j}) &= w(f|g_\text{cv} f_{\text{cut},j} / 2)
    \\
    w_{\text{right}}(f|f_{\text{cut},j}) &= w(f|g_\text{cv} f_{\text{cut},j}) - w_{\text{left}}(f|f_{\text{cut},j})
$$

The parameter $g_\text{cv}$ is a user-defined parameter to control the amount of data used in the refinements.
In STACIE, the default value is $g_\text{cv} = 1.25$,
meaning that 25% more data is used compared to the original fit.
(This makes the cross-validation a bit more sensitive to underfitting,
which we found to be beneficial in practice.)
This parameter can be controlled with the option `fcut_factor`
of the [`CV2LCriterion`](#stacie.cutoff.CV2LCriterion) class.
An instance of this class can be passed to the `cutoff_criterion` argument
of the function [`estimate_acint()`](#stacie.estimate.estimate_acint).

Instead of performing two full non-linear regressions of the parameters for the two halves,
we use the linear approximation of the changes in parameters.
For cutoffs leading to well-behaved fits, these corrections are small,
justifying the use of a linear approximation.

The design matrix of the linear regression is:

$$
    D_{kp} = \left.
            \frac{
                \partial I^\text{model}(f_k; \mathbf{b})
            }{
                \partial b  _p
            }
        \right|_{\mathbf{b} = \hat{\mathbf{b}}^{(j)}}
$$

The expected values are the residual between the sampling PSD and the model:

$$
    y_k = \hat{\mathbf{I}}_k - \mathbf{I}^\text{model}(f_k; \hat{\mathbf{b}}^{(j)})
$$

The measurement error is the standard deviation of the Gamma distribution,
using the model spectrum in the scale parameter and the shape parameter of the smapling PSD:

$$
    \sigma_k = \frac{\mathbf{I}^\text{model}(f_k; \hat{\mathbf{b}}^{(j)})}{\sqrt{\alpha_k}}
$$

The weighted regression to obtain first-order corrections to the parameters $\hat{\mathbf{b}}^{(j)}$
solves the following linear system in the least-squares sense:

$$
    \frac{w_k}{\sigma_k} \sum_{p=1}^{P} \hat{b}^{(j)}_{\text{corr},p} D_{kp} = \frac{w_k}{\sigma_k} y_k
$$

where $w_k$ is the weight of the $k$-th frequency point.
This system is solved once with weights for the left half and once for the right half.

The function [`linear_weighted_regression()`](#stacie.cutoff.linear_weighted_regression)
has a robust pre-conditioned implementation of the above linear regression.
It can handle multiple weight vectors at once,
and can directly compute linear combinations of parameters for different weight vectors.
It is used to directly compute the difference between the corrections for the left and right halves,
denoted as $\hat{\mathbf{d}}$, and its covariance matrix $\hat{\mathbf{C}}_{\mathbf{d},\mathbf{d}}$.
Normally, the model parameters fitted to both halves must be the same,
and the negative log-likelihood that the fitted parameters are indeed identical is given by:

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
As the cutoff frequency and the effective number of fitting points increases,
the model becomes better constrained.
The second term will decrease but as soon as the model underfits the data,
the third term (the bias term) will steeply increase.
Practically, the cutoff scan is interrupted when the criterion exceeds the incumbent by $g_\text{incr}$.
The default value is $g_\text{incr}=100$, but this can be changed
with the option `criterion_high` of the function [`estimate_acint()`](#stacie.estimate.estimate_acint).

A good cutoff frequency is the one that minimizes the criterion and thereby finds a good compromise
between bias and variance.

## Marginalization over the cutoff frequency

Any method to deduce the cutoff frequency from the spectrum,
whether it is human judgment or an automated algorithm,
will introduce some uncertainty in the final result,
because the spectrum with some statistical uncertainty is used as input.

In STACIE, we account for this uncertainty by marginalizing the model parameters over the cutoff frequency,
using $\mathcal{L}^\text{CV2L}$ as a model for the likelihood.
This naturally accounts for the uncertainty in the cutoff frequency
and is preferred over fixing the cutoff frequency at a single value.

Practically, the final estimate of the parameters and their covariance is computed
using [standard expression for mixture distributions](https://en.wikipedia.org/wiki/Mixture_distribution#Moments):

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
The weights $W_j$ sum up to 1 and are proportional to $\mathcal{L}^\text{CV2L}$.
