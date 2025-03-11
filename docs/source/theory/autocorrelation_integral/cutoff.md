# Frequency Cutoff

The Exponential Tail model should be fitted to the low-frequency part of the spectrum,
but it is not clear *a priori* up to which frequency.
Stacie's strategy is to test several frequency cutoffs, uniformly distributed on a log scale.
The best cutoff is identified as the one
that maximizes the amount of data included without an obvious trend in the residuals.
Including more data points reduces the variance of the autocorrelation integral estimate,
but any trend in the residuals would be a sign of underfitting, leading to biased parameters.

Visually recognizing underfitting may seem trivial,
but designing a robust criterion to quantify it is not.
Traditional model selection criteria such as
[Marginal Likelihood](https://en.wikipedia.org/wiki/Marginal_likelihood),
[Akaike's Information Criterion (AIC)](https://en.wikipedia.org/wiki/Akaike_information_criterion),
or [Bayesian Information Criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion)
are not trivially applicable because they solve a different problem:
identifying the best model for a given data set.
Here, the model is the same for all cutoffs, but the amount of fitting data varies.

A new "underfitting criterion" (UFC) for selecting the best cutoff frequency is presented below.
Its value decreases slowly as more data is included and as long as the residuals exhibit no trends.
As soon as any trend appears in the residuals, the underfitting criterion increases sharply.
The amount of data for which the criterion becomes minimal represents a sweet spot:
the amount of useful information is maximized,
subject to the constraint that there is no risk of biasing the parameters.

The UFC and AIC (with a workaround discussed below) can be used to select a domain
when fitting a smooth univariate model to noisy data.
Both criteria are illustrated for a simple polynomial fit to noisy cosine data
in the [Underfitting Criterion Demo](../../examples/underfitting_criterion.py).
This example may be helpful to gain some intuition for the theory derived below.

## Normalized Residuals

The underfitting criterion is applicable to any 1D curve fitting problem
where only a part of the data is expected to follow the model.
This "part" is not known *a priori*,
and an appropriate range of the dependent variables is searched.
For each selected range, the model is fitted to $N_\text{fit}$ data points.
The corresponding data points are labeled with an integer index $i \in \{1 \ldots N_\text{fit}\}$.
The index must preserve the order of the independent variables.
For example, in the case of a spectrum, from low frequency to high frequency.

The underfitting criterion uses normalized residuals after maximizing the likelihood:

$$
    \hat{R}_i = \frac{\hat{Y}_i - \hat{\mu}_i}{\hat{\sigma}_i}
$$

where $\hat{\mu}_i$ and $\hat{\sigma}_i$
are the maximum-likelihood mean and standard deviation of the model for data point $i$.
In the case of spectra, the residuals in the notation from the
[Parameter Estimation](../model.md#lmax-target) section are:

$$
  \hat{Y}_i &= \hat{C}_i
  \\
  \hat{\mu}_i &= \kappa_i\,\hat{\theta}_i
  \\
  \hat{\sigma}_i &= \sqrt{\kappa_i}\,\hat{\theta}_i
$$

The parameter $\hat{\theta}_i$ is the scale of the Gamma distribution at each frequency,
found with the best parameters, i.e. those that maximize the log-likelihood.

In a regression problem (without underfitting), we would have the following expectation values:

$$
    \mean[\hat{R}_i] = 0 \qquad \mean[\hat{R}^2_i] < 1
$$

Although the residuals have been normalized, their variance should be less than one.
On average,
the maximum likelihood model is (slightly) closer to the measured data than to the ground truth
because it tries to make the residuals as small as possible.
In other words, on average, likelihood maximization will result in a slightly overfitted model.

If a model is too simple to describe the data,
there will be some underfitting and the normalized residuals can become (much) larger than one.
Therefore, in principle, a sample variance of the normalized residuals
greater than one can be used to detect underfitting.
However, such a simple criterion does not detect small misfits.
In edge cases, the variance of the normalized residuals may still be close to one,
even though they show a trend.

## Cumulative Sums of Residuals

Underfitting can be detected more robustly by
making use of correlations between nearby residuals.
For example, the analytical spectrum of the Exponential Tail model is smooth
compared to the uncorrelated uncertainties of a power spectrum.
If the model cannot fully describe the data,
nearby residuals will have the same sign on average.

One can amplify correlations between nearby residuals
by using a shifted cumulative sum of residuals:

$$
    \hat{U}_j
        = 2\sum_{i=1}^j \hat{R}_i - \sum_{i=1}^{N_\text{fit}} \hat{R}_i
           \qquad \forall\,j \in \lbrace 0, \ldots, N_\text{fit}\rbrace
$$

The first term is twice the cumulative sum of the residuals,
and the second term shifts them down by the sum of all residuals.
This shifted cumulative sum can be rewritten as:

$$
    \hat{U}_j
        = \sum_{i=1}^j \hat{R}_i - \sum_{i=j+1}^{N_\text{fit}} \hat{R}_i
$$

In this form, $\hat{U}_j$ is the sum of all residuals up to index $j$,
minus the sum of the remaining residuals.
If $j$ corresponds to the data point where the model goes from over- to undershooting the data,
then $\hat{U}_j$ will add systematic errors on both sides of $j$ with the same sign.

In the absence of underfitting, the corresponding expectation values should satisfy:

$$
    \mean[\hat{U}_j] = 0 \qquad \mean[\hat{U}^2_j] < N_\text{fit}
$$

However, in case of underfitting, there is a trend in the residuals,
meaning that nearby residuals are positively correlated.
In this case, the positively correlated errors will add up in the cumulative sums,
such that the sample variance of $\hat{U}^2_j$ quickly exceeds the expectation value above.

## Underfitting Criterion

We propose the following criterion to detect underfitting artifacts:

$$
    \operatorname{UFC}
        = \frac{1}{N_\text{fit} + 1}
          \left(\sum_{j=0}^{N_\text{fit}} \hat{U}^2_j\right)
          - N_\text{fit}
$$

Averaging over all indexes $j$ in the first term has the advantage that the UFC accounts for
all possible indexes $j$ where systematic errors may change sign.

If there is no underfitting, the UFC will be negative.
Empirically, we observe that the UFC decreases slowly with increasing amounts of data,
e.g., increasing frequency cutoff,
as long as there is no underfitting.
As soon as the model begins to underfit the data,
the sample variance of the cumulative sums and the UFC increase rapidly.
The frequency cutoff for which the UFC is minimal
always seems to strike a good balance between over- and underfitting.
The minimum is located at the onset of underfitting,
which is barely visible in the residuals to the naked eye.
At this point, one has (nearly) the maximum amount of data in the fit
(and thus minimal uncertainty of the parameters)
without biasing the estimated parameters due to underfitting.

## Akaike's Information Criterion

Akaike's Information Criterion (AIC) is intended for model selection, not data selection.
At first glance, it does not seem applicable to frequency cutoff selection.
However, it is possible reformulate data selection as model selection.

The "trick" is to introduce an additional dummy parameter to "fit" the discarded data point.
By introducing such dummy parameters,
all data in the spectrum is always included in the likelihood (regardless of the cutoff).
The cutoff simply determines at which part is modeled
with the Exponential Tail model or the dummy parameters.
Finding the optimal cutoff is thus reduced to a model selection problem.

In the case of the frequency cutoff,
the dummy parameter corresponds to the scale parameter of the Gamma distribution.
Assuming that $\kappa_k > 1$, the probability for this point is maximized by:

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
Nevertheless, in our numerical benchmarks,
the frequency cutoff that minimizes the AIC always leads to visually clearly biased fits.
In contrast, a cutoff that minimizes the UFC always leads to residuals without a noticeable trend.

We must emphasize that the UFC is an ad hoc construction.
It is nothing more than a robust mathematical implementation
of how one would intuitively describe the underfitting of a smooth 1D curve to noisy data.
Nevertheless, its use is well justified by its practical utility,
its simplicity, its effectiveness in our benchmarks,
and the complete absence of manually tunable settings.
