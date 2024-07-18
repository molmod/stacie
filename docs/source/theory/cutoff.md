# Frequency Cutoff

The Exponential Tail model should be fitted to the low-frequency part of the spectrum,
but it is not clear *a priori* up to which frequency.
Stacie's strategy is to test several frequency cutoffs, uniformly distributed on a log scale.
The best cutoff is identified as the one
that maximizes the amount of included data without an apparent trend the residuals.
Any trend in the residuals would be a sign of underfitting, leading to biased parameters.

Visually recognizing underfitting may seem trivial,
but designing a robust criterion to quantify it is not.
Traditional model selection criteria,
such as [Marginal Likelihood](https://en.wikipedia.org/wiki/Marginal_likelihood),
[Akaike's Information Criterion (AIC)](https://en.wikipedia.org/wiki/Akaike_information_criterion),
or [Bayesian Information Criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion)
are not trivially applicable because they solve a different problem:
they identify the best model for a given data set.
Here, the model is the same for all cutoffs, but the amount of fitting data varies.

A new "underfitting criterion" (UFC) for selecting the best cutoff frequency is presented below.
Its value slowly decreases as more data is included and as long as the residuals exhibit no trends.
As soon as any trend appears in the residuals, the underfitting criterion sharply increases.
The amount of data for which the criterion becomes minimal represents a sweet spot:
the amount of useful information is maximized,
subject to the constraint that there is no risk of biasing the parameters.

Both AIC and UFC can be used for the selection of a domain
when fitting a smooth univariate model to noisy data.
They is illustrated for a simple polynomial fit to noisy cosine data
in the [Underfitting Criterion Demo](../examples/underfitting_criterion.py).
This example may be helpful to gain some intuition for the theory derived below.


## Normalized Residuals

The underfitting criterion, uses normalized residuals after maximization of the likelihood:

$$
    \hat{R}_k = \frac{\hat{Y}_k - \hat{\mu}_k}{\hat{\sigma}_k}
$$

where $\hat{\mu}_k$ and $\hat{\sigma}_k$ are the maximum-likelihood mean and standard deviation of the model for data point $k$. In the case of spectra, these required quantities, using the from the [Parameter Estimation](#lmax-target) section, are:

$$
  \hat{Y}_k &= \hat{C}_k
  \\
  \hat{\mu}_k &= \kappa_k\,\hat{\theta}_k
  \\
  \hat{\sigma}_k &= \sqrt{\kappa_k}\,\hat{\theta}_k
$$

The parameter $\hat{\theta}_k$ is the scale of the Gamma distribution at each frequency,
found with the best parameters, i.e. those that maximize the log-likelihood.

In the remainder of this section, we will relabel the indices $k$ of the residuals
with an integer $i$ going from $1$ to $N_\text{res}$
maintaining the same order as the index $k$ (from low to high frequency).

In a regression problem (without underfitting), we would have the following expectation values:

$$
    \mean[\hat{R}_i] = 0 \qquad \mean[\hat{R}^2_i] < 1
$$

Despite having normalized the residuals, their variance should be less than one.
On average, the maximum likelihood model will be (slightly) closer to the measurement data
than the ground truth, because it tries to make the residuals as small as possible.

When a model is too simple to describe the data,
there will be some underfitting and the normalized residuals can become (much) larger than 1.
Therefore, in principle, a sampling variance of the normalized residuals
greater than 1 can be used to detect underfittng.
However, such a simple criterion does not easily reveal slight misfits.
In edge cases, the variance of the normalized residuals can still be about 1,
despite the fact that they exhibit a trend.
Such a simple criterion is therefore not suitable for selecting an appropriate cutoff.


## Cumulative Sums of Residuals

Underfitting can be detected more robustly by
making use of correlations between nearby residuals.
The analytical spectrum of the Exponential Tail model is smooth
compared to the uncorrelated uncertainties of a power spectrum.
When the model cannot fully describe spectral data,
nearby residuals will, on average, have the same sign.

One can amplify correlations between nearby residuals
by using a shifted cumulative sum of residuals:

$$
    \hat{U}_j
        &= 2\sum_{i=1}^j \hat{R}_i - \sum_{i=1}^{N_\text{res}} \hat{R}_i
           \qquad \forall\,j \in \lbrace 0, \ldots, N_\text{res}\rbrace
    \\
        &= \sum_{i=1}^j \hat{R}_i - \sum_{i=j+1}^{N_\text{res}} \hat{R}_i
$$

$\hat{U}_j$ can also be interpreted as the sum of all residuals up to index $j$,
minus the sum of the remaining residuals.
If $j$ corresponds to the data point where the model goes from over- to undershooting the data,
then the corresponding $\hat{U}_j$ will add systematic errors on both sides of $j$ with the same sign.

In the absence of underfitting, the corresponding expectation values should satisfy:

$$
    \mean[\hat{U}_j] = 0 \qquad \mean[\hat{U}^2_j] < N_\text{res}
$$

However, in case of underfitting, there is a trend in the residuals,
meaning that nearby residuals are positively correlated.
In this case, the positively correlated errors will add up in the cumulative sums,
such that the empirical variance of $\hat{U}^2_j$ quickly exceeds the expectation value above.


## Underfitting Criterion

We propose the following criterion to detect underfitting artifacts:

$$
    \operatorname{UFC}
        = \frac{1}{N_\text{res} + 1} \sum_{j=0}^{N_\text{res}}
           \left(\hat{U}^2_j\right) - N_\text{res}
$$

Averaging over all indexes $j$ has the advantage that
all possible locations $j$ where systematic errors change sign are taken into account.

Before there are any signs of underfitting, the UFC is negative.
Empirically, we always observe that the UFC slowly decreases with increasing amounts of data,
as long as there is no underfitting.
As soon as the model underfits the data,
the sampling variance of the cumulative sums will quickly exceed
the upper limit of the expectation value and the UFC will increase rapidly.

We observed empirically that the frequency cutoff for which the UFC is minimal
always strikes a good balance between over- and underfitting.
The minimum is always located at the onset of underfitting,
which is barely visible in the residuals to the naked eye.
At this point, one has (nearly) the maximum amount of data in the fit
(and thus minimal uncertainty in the parameters)
without biasing the estimated parameters due to underfitting.


## Akaike's Information Criterion

Akaike's Information Criterion (AIC) is intended for model selection, not data selection,
so, at first glance, it does not seem to be applicable to the cutoff selection.
However, it is possible reformulate data selection as model selection.

The "trick" is to introduce one extra dummy parameter for every discarded data point.
The dummy parameter is simply the model value for that data point and will therefore coincide
with the measurement value.
By introducing these dummy parameters,
all data in the spectrum are always included in the likelihood (irrespective of the cutoff).
The cutoff just determines at which part is modeled with the Exponential Tail model or the dummy
parameters.
Finding the optimal cutoff is therefore reduced to a model selection problem.

In the case of the frequency cutoff,
the dummy parameter corresponds to the scale parameter of the Gamma distribution.
Assuming that $\kappa_k > 1$, the likelihood for that point is maximized by:

$$
    \hat{\theta}_k = \frac{\hat{C}_k}{\kappa_k - 1}.
$$

The log likelihood for such a point becomes:

$$
    \ln \mathcal{L}_k^\text{cut} =
        -\ln\Gamma(\kappa_k)
        - \ln\left(\frac{\hat{C}_k}{\kappa_k - 1}\right)
        + (\kappa_k - 1) \Bigl(\ln(\kappa_k - 1) - 1\Bigr)
$$

To obtain the total log likelihood, $\ln\mathcal{L}^\text{total}$,
the terms $\ln \mathcal{L}_k^\text{cut}$ can be summed over all discarded points
and added to the likelihood of the model fitted to the data before the cutoff.
The AIC with this likelihood then becomes:

$$
    \operatorname{AIC}
    = 2 (N_\text{par} + N_\text{discard}) - 2 \ln\mathcal{L}^\text{total}
$$

where:

- $N_\text{par}$ is the number of parameters in the model for the low-frequency part of the spectrum
  (3 in case of the Exponential Tail model),
- $N_\text{discard}$ is the number of discarded data points


## Cutoff selection

The AIC is the more fundamental of the two criteria and it is also widely used.
Nevertheless, in our numerical benchmarks, the AIC always leads to visually clearly biased fits,
while the UFC systematically selects cutoffs for which the residuals are featureless.

We must emphasize that the UFC is an ad hoc construction.
It is nothing more than a robust mathematical implementation
of how one would intuitively describe underfitting of a smooth model to noisy data.
Nevertheless, its use is well justified by its practical utility,
its simplicity, its effectiveness in our benchmarks,
and the complete absence of manually tunable settings.
