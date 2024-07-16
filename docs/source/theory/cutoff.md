# Frequency Cutoff

The model spectrum should be fitted to the low-frequency spectrum,
but it is not clear *a priori* up to which frequency data should be included in the fit.
Stacie's strategy is to test several frequency cutoffs, uniformly distributed on a log scale,
and determine the best cutoff that minimizes the risk for over- or underfitting.
This may sound straightforward, but designing a metric for this risk is not trivial.
Traditional model selection criteria, such as evidence in Bayesian inference,
[Akaike's Information Criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion),
or [Bayesian Information Criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion)
attempt to identify the best model for a given data set.
They solve a different problem and are not (trivially) applicable.
Here, the model is the same for all cutoffs, but the amount of fitting data varies.

## Normalized Residuals

The risk metric proposed here, uses normalized residuals, which are defined as follows,
using the [likelihood maximization](#lmax-target) notation:

$$
  \hat{R}_k = \frac{\frac{\hat{C}_k}{\hat{\theta}_k} - \kappa_k}{\sqrt{\kappa_k}}
$$

The parameter $\hat{\theta}_k$ is the scale of the Gamma distribution at each frequency data point,
as found with the best parameters, i.e. those that maximize the log-likelihood.

In the remainder of this section, we will relabel the indices of the residuals
with an index $i$ going from $1$ to $N_\text{res}$
maintaining the same order as the index $k$ (from low to high frequency).

If the empirical spectrum were sampled
from the Gamma distribution predicted by the Exponential Tail model,
we would have the following expectation values:

$$
    \mean[\hat{R}_i] = 0 \qquad \mean[\hat{R}^2_i] = 1
$$

With real data and model parameters fitted to those data, this will no longer hold in general,
primarily due to overfitting (too little data)
or underfitting (not all data can be explained by the model).
However, a direct analysis of the distribution of the normalized residuals, $\hat{R}_i$,
does not easily reveal slight misfits:
Individual residuals can all contribute to a high likelihood,
as long as the underfitting is not too severe.
This hampers the selection of an appropriate cutoff with model selection criteria,
not only because they are designed for a different purpose,
but also because they treat each residual independently.

## Cumulative Sums of Residuals

A more sensitive metric for detecting over- or underfitting artifacts
makes use of correlations between nearby residuals.
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

One can interpret $\hat{U}_j$ as the sum of all residuals up to index $j$,
followed by the sum of the remaining residuals with a change of sign.
If $j$ corresponds to the data point where the model goes from over- to undershooting the data,
then the corresponding $\hat{U}_j$ will add systematic errors on both sides of $j$ with the same sign.

The corresponding expectation values,
if the residuals were sampled from the model distribution, are:

$$
    \mean[\hat{U}_j] = 0 \qquad \mean[\hat{U}^2_j] = N_\text{res}
$$

If there is even a small trend in the residuals,
the correlated errors will add up in the cumulative sums,
leading to pronounced deviations between the above expectation values and their empirical estimates.

## Proposed Risk Metric and Cutoff Selection

We propose the following risk metric to detect over- or underfitting artifacts:

$$
    \mathcal{M}
        &= \frac{1}{N_\text{res} + 1} \sum_{j=0}^{N_\text{res}}
           \left(\hat{U}^2_j - \mean[\hat{U}^2_j]\right)
    \\
        &= \left(\frac{1}{N_\text{res} + 1} \sum_{j=0}^{N_\text{res}} \hat{U}^2_j\right) -
           N_\text{res}
$$

When the model underfits the data,
the sampling variance of the cumulative sums will quickly exceed the corresponding expectation value
and the metric $\mathcal{M}$ will become very large.
Frequency cutoffs for which $\mathcal{M}$ becomes large can be safely discarded.
Averaging over all indexes $j$ has the advantage that
all possible locations $j$ where systematic errors change sign are taken into account.

We found that the frequency cutoff that minimizes $\mathcal{M}$
always strikes a good balance between over- and underfitting.
This can be understood as follows:
As long as the frequency cutoff is low enough to avoid underfitting,
the variance of the normalized residuals is slightly less than $1$
(because there are three degrees of freedom in the fit).
This small discrepancy causes $\mathcal{M}$ to decrease slowly as more data points are added,
as long as the model does not exhibit underfitting.
At the onset of underfitting, which is barely visible to the naked eye,
$\mathcal{M}$ increases rapidly.
At this point, one has included (nearly) the maximum amount of data in the fit
(and thus minimal uncertainty in the parameters)
without biasing the estimated parameters due to underfitting.


Finally, we must emphasize that the proposed risk metric $\mathcal{M}$ is an ad hoc construction.
It is nothing more than a robust mathematical implementation
of how one would intuitively describe underfitting of a smooth model to noisy data.
Nevertheless, its use is well justified by its practical utility,
its simplicity, its effectiveness in our benchmarks,
and the complete absence of manually tunable settings.
