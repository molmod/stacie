# Parameter Estimation

Before discussing how to fit a model to spectral data,
we first review the statistics of the sampling {term}`PSD`.
Given these statistical properties,
we can derive the likelihood that certain model parameters explain the observed PSD.

## Statistics of the Sampling Power Spectral Distribution

When constructing an estimate of a discrete PSD from a finite amount of data,
it is bound to contain some {term}`uncertainty`, which will be characterized below.

The estimate of the PSD is sometimes also called
the [periodogram](https://en.wikipedia.org/wiki/Periodogram) or the (empirical) power spectrum.

Consider a periodic random real sequence $\hat{\mathbf{x}}$ with elements $\hat{x}_n$ and period $N$.
For practical purposes, it is sufficient to consider one period of this infinitely long sequence.
The mean of the sequence is zero, and its covariance is $\cov[\hat{x}_n \,,\, \hat{x}_m]$.
The distribution of the sequence is stationary,
i.e., each time translation of a sequence results in an equally probable sample.
As a result, the covariance has a circulant structure:

$$
    \cov[\hat{x}_n \,,\, \hat{x}_m] = c_\Delta = c_{-\Delta}
$$

with $\Delta=n-m$.
Thus, we can express the covariance with a single index and treat it as a real periodic sequence,
albeit not stochastic.
$c_\Delta$ is also known as the autocovariance or autocorrelation function of the stochastic process
because it expresses the covariance of a sequence $\hat{\mathbf{x}}$
with itself translated by $\Delta$ steps.

The discrete Fourier transform of the sequence is:

$$
    \hat{X}_k = \sum_{n=0}^{N-1} \hat{x}_n \omega^{-kn}
$$

with $\omega = e^{2\pi i/N}$.

A well-known property of circulant matrices is that their eigenvectors
are sine- and cosine-like basis functions.
As a result, the covariance of the discrete Fourier transform $\hat{\mathbf{X}}$ becomes diagonal.
To make this derivation self-contained, we write out the mean and covariance of $\hat{X}_k$ explicitly.
Note that the operators $\mean[\cdot]$, $\var[\cdot]$, and $\cov[\cdot,\cdot]$
are expected values over all possible realizations of the sequence.

For the expected value of the Fourier transform,
we take advantage of the fact that all time translations of $\hat{\mathbf{x}}$
belong to the same distribution.
We can explicitly compute the average over all time translations,
in addition to computing the mean, without loss of generality.
In the last steps, the index $n$ is relabeled to $n-m$, and some factors are rearranged,
after which the sums can be worked out.

$$
    E[\hat{X}_k]
        &= \mean\left[
            \sum_{n=0}^{N-1} \hat{x}_n \omega^{-kn}
        \right]
        \\
        &= \mean\left[
            \frac{1}{N} \sum_{m=0}^{N-1}\sum_{n=0}^{N-1} \hat{x}_{n+m} \omega^{-kn}
        \right]
        \\
        &= \mean\left[
            \frac{1}{N}
            \underbrace{\left(\sum_{m=0}^{N-1} \omega^{km}\right)}_{=0}
            \sum_{n=0}^{N-1} \hat{x}_{n} \omega^{-kn}
        \right]
        \\
        &= 0
$$

The derivation of the covariance uses similar techniques.
In the following derivation, $*$ stands for complex conjugation.
Halfway through, the summation index $n$ is written as $n=\Delta+m$.

$$
    \cov[\hat{X}^*_k\,,\,\hat{X}_\ell]
    &= \cov\left[
        \sum_{m=0}^{N-1} \hat{x}_m \omega^{km}
        \,,\,
        \sum_{n=0}^{N-1} \hat{x}_n \omega^{-\ell n}
    \right]
    \\
    &= \sum_{m=0}^{N-1} \sum_{n=0}^{N-1} \omega^{km-\ell n} c_{n-m}
    \\
    &= \sum_{m=0}^{N-1} \omega^{km-\ell m}\, \sum_{\Delta=0}^{N-1} \omega^{-\ell\Delta} c_\Delta
    \\
    &= N\delta_{k,\ell} \,\mathcal{F}[c]_\ell
$$

To finalize the result,
we need to work out the discrete Fourier transform of the autocorrelation function, $c_\Delta$.
Again, we make use of the freedom to insert a time average when computing a mean.
Note that this derivation assumes $\mean[\hat{x}_n]=0$ to keep the notation bearable.

$$
    C_k = \mathcal{F}[\mathbf{c}]_k
    &= \sum_{\Delta=0}^{N-1} \omega^{-k\Delta} \mean\left[
        \frac{1}{N}
        \sum_{n=0}^{N-1}\hat{x}_n\, \hat{x}_{n+\Delta}
    \right]
    \\
    &= \frac{1}{N} \mean\left[
        \sum_{n=0}^{N-1}\omega^{kn}\hat{x}_n\,
        \sum_{\Delta=0}^{N-1}\omega^{-k\Delta-kn} \hat{x}_{n+\Delta}
    \right]
    \\
    &= \frac{1}{N} \mean\Bigl[|\hat{X}_k|^2\Bigr]
$$

This is the discrete version of the Wiener--Khinchin theorem {cite:p}`oppenheim_1999_power`.

By combining the previous two results,
we can write the covariance of the Fourier transform of the input sequence as:

$$
    \cov[\hat{X}^*_k \,,\, \hat{X}_\ell]
    = \delta_{k,\ell} \mean\Bigl[|\hat{X}_k|^2\Bigr]
    = N \delta_{k,\ell} C_k
$$

For the real component of $\hat{X}_k$ $(=\hat{X}^*_{-k})$, we find:

$$
    \var[\Re (\hat{X}_k)]
    &= \frac{1}{4}\var[\hat{X}_k + \hat{X}^*_k]
    \\
    &= \frac{1}{4}\Bigl(
        \cov[\hat{X}_k \,,\, \hat{X}_k]
        + \cov[\hat{X}_k \,,\, \hat{X}^*_k]
        + \cov[\hat{X}^*_k \,,\, \hat{X}_k]
        + \cov[\hat{X}^*_k \,,\, \hat{X}^*_k]
    \Bigr)
    \\
    &= \frac{1}{4}\Bigl(
        \cov[\hat{X}^*_{-k} \,,\, \hat{X}_k]
        + \cov[\hat{X}_k \,,\, \hat{X}^*_k]
        + \cov[\hat{X}^*_k \,,\, \hat{X}_k]
        + \cov[\hat{X}^*_k \,,\, \hat{X}_{-k}]
    \Bigr)
    \\
    &= \begin{cases}
        N C_0 & \text{if } k=0 \\
        \frac{N}{2} C_k & \text{if } 0<k<N
    \end{cases}
$$

Similarly, for the imaginary component (which is zero for $k=0$):

$$
    \var[\Im (\hat{X}_k)]
    &= \frac{1}{4}\var[\hat{X}_k - \hat{X}^*_k]
    \\
    &= \frac{1}{4}\Bigl(
        \cov[\hat{X}_k \,,\, \hat{X}_k]
        - \cov[\hat{X}_k \,,\, \hat{X}^*_k]
        - \cov[\hat{X}^*_k \,,\, \hat{X}_k]
        + \cov[\hat{X}^*_k \,,\, \hat{X}^*_k]
    \Bigr)
    \\
    &= \begin{cases}
        0 & \text{if } k=0 \\
        \frac{N}{2} C_k & \text{if } 0<k<N
    \end{cases}
$$

The real and imaginary components have no covariance:

$$
    \cov[\Re (\hat{X}_k)\,,\Im (\hat{X}_k)]
    &= \frac{1}{4}\cov[\hat{X}_k + \hat{X}^*_k \,,\, \hat{X}_k - \hat{X}^*_k]
    \\
    &= \frac{1}{4}\Bigl(
        \cov[\hat{X}_k \,,\, \hat{X}_k]
        - \cov[\hat{X}_k \,,\, \hat{X}^*_k]
    \\
    &\qquad\qquad + \cov[\hat{X}^*_k \,,\, \hat{X}_k]
        - \cov[\hat{X}^*_k \,,\, \hat{X}^*_k]
        \Bigr)
    \\
    &= 0
$$

In summary, the Fourier transform of a stationary stochastic process
consists of uncorrelated real and imaginary components at each frequency.
Furthermore, the variance of the Fourier transform is proportional to the power spectrum.
This simple statistical structure makes the spectrum a convenient starting point
for further analysis and uncertainty quantification.
In comparison, the ACF has non-trivial correlated uncertainties
{cite:p}`bartlett_1980_introduction,boshnakov_1996_bartlett,francq_2009_bartlett`,
making it difficult to fit models directly to the ACF (or its running integral).

If we further assume that the sequence $\hat{\mathbf{x}}$ is the result of a periodic Gaussian process,
the Fourier transform is normally distributed.
In this case, the empirical power spectrum follows a scaled Chi-squared distribution
{cite:p}`priestley_1982_spectral, fuller_1995_introduction, shumway_2017_time, ercole_2017_accurate`.
For notational consistency, we will use the
[$\gdist(\alpha,\theta)$ distribution](https://en.wikipedia.org/wiki/Gamma_distribution)
with shape parameter $\alpha$ and scale parameter $\theta$:

$$
    \hat{C}_0=\frac{1}{N}|\hat{X}_0|^2
    &\sim \gdist(\textstyle\frac{1}{2},2C_0)
    \\
    \hat{C}_{N/2}=\frac{1}{N}|\hat{X}_{N/2}|^2
    &\sim \gdist(\textstyle\frac{1}{2},2C_{N/2})
    \quad \text{if $N$ is even}
    \\
    \hat{C}_k=\frac{1}{N}|\hat{X}_k|^2
    &\sim \gdist(1,C_k)
    \quad \text{for } 0<k<N \text { and } k \neq N/2
$$

Note that $\hat{X}_0$ and $\hat{X}_{N/2}$ have only a real component
because the input sequence $\hat{\mathbf{x}}$ is real,
which corresponds to a Chi-squared distribution with one degree of freedom.
For all other frequencies, $\hat{X}_k$ has a real and imaginary component,
resulting in two degrees of freedom.

Spectra are often computed by averaging them over $M$ sequences to reduce the variance.
In this case, the $M$-averaged empirical spectrum is distributed as:

$$
    \hat{C}_k=\frac{1}{NM}\sum_{s=1}^M|\hat{X}^s_0|^2
    \sim \gdist(\textstyle\frac{\nu_k}{2},\textstyle\frac{2}{\nu_k}C_k)
$$

with

$$
    \nu_k = \begin{cases}
        M & \text{if $k=0$} \\
        M & \text{if $k=N/2$ and $N$ is even} \\
        2M & \text{otherwise}
    \end{cases}
$$

The rescaled spectrum used in STACIE, $\hat{I}_k$, has the same distribution,
except for the scale parameter:

$$
    \hat{I}_k = \frac{F h}{2} \hat{C}_k
    \sim \gdist(\textstyle\frac{\nu_k}{2},\textstyle\frac{2}{\nu_k}I_k)
$$

(lmax-target)=

## Regression

To identify the low-frequency part of the spectrum,
we introduce a smooth switching function that goes from 1 to 0 as the frequency increases:

$$
    w(f_k|f_\text{cut}) = \frac{1}{1 + (f_k/f_\text{cut})^\beta}
$$

This switching function is $1/2$ when $f_k=f_\text{cut}$.
The hyperparameter $\beta$ controls the steepness of the transition and is 8 by default.
(This should be fine for most applications.)
This value can be set with the `switch_exponent` argument
of the [estimate_acint()](#stacie.estimate.estimate_acint) function.
One can better appreciate the advantage of this switching function by rewriting with a hyperbolic tangent:

$$
    w(f_k|f_\text{cut}) = \frac{1}{2}\left[
        1 - \tanh\left(\frac{\beta}{2}\ln\frac{f_k}{f_\text{cut}}\right)
    \right]
$$

This shows that the switching is scale invariant, i.e., it does not depend on the unit of the frequency,
because the frequency appears only in a logarithm.
The parameter $\beta$ controls the width of the transition region on a logarithmic scale.

We derive below how to fit parameters for a given frequency cut-off $f_\text{cut}$.
The [next section](cutoff.md) describes how to find suitable cutoffs.

To fit the model, we use a form of local regression
by introducing weights into the log-likelihood function.
The weighted log likelihood of the model $I^\text{model}_k(\mathbf{b})$
with parameter vector $\mathbf{b}$ becomes:

$$
    \ln\mathcal{L}(\mathbf{b})
    &=\sum_{k\in K} w(f_k|f_\text{cut}) \ln p_{\gdist(\alpha_k,\theta_k)}(\hat{C}_k)
    \\
    &=\sum_{k\in K}
        w(f_k|f_\text{cut}) \left[
            -\ln \Gamma(\alpha_k)
            - \ln\bigl(\theta_k(\mathbf{b})\bigr)
            + (\alpha_k - 1)\ln\left(\frac{\hat{C}_k}{\theta_k(\mathbf{b})}\right)
            - \frac{\hat{C}_k}{\theta_k(\mathbf{b})}
        \right]
$$

with

$$
    \alpha_k &= \frac{\nu_k}{2}
    \\
    \theta_k(\mathbf{b}) &= \frac{2 I^\text{model}_k(\mathbf{b})}{\nu_k}
$$

This log-likelihood is maximized to estimate the model parameters.
The zero-frequency limit of the fitted model is then the estimate of the autocorrelation integral.

:::{note}
It is worth mentioning that the cutoff frequency is not a proper hyperparameter in the Bayesian sense.
It appears in the weight factor $w(f_k|f_\text{cut})$, which is not part of the model.
Instead, it is a concept taken from local regression methods.
One conceptual limitation of this approach is that the unit of the likelihood function,
$\mathcal{L}(\mathbf{b})$, depends on the cutoff frequency.
As a result, one cannot compare the likelihood of two different cutoffs.
This is of little concern when fitting parameters for a fixed cutoff,
but it is important to keep in mind when searching for suitable cutoffs.
:::

For compatibility with the SciPy optimizers,
the cost function $\operatorname{cost}(\mathbf{b}) = -\ln \mathcal{L}(\mathbf{b})$ is minimized.
STACIE implements first and second derivatives of $\operatorname{cost}(\mathbf{b})$,
and also a good initial guess of the parameters, using efficient vectorized NumPy code.
These features make the optimization of the parameters both efficient and reliable.
The optimized parameters are denoted as $\hat{\mathbf{b}}$.
The hats indicate that these are statistical estimates because
they are derived from data statistical uncertainties.

The Hessian computed with the estimated parameters, $\operatorname{cost}(\hat{\mathbf{b}})$,
must be positive definite.
(If non-positive eigenvalues are found, the optimization is treated as failed.)

$$
    \hat{\mathbf{H}} > 0 \quad \text{with}
    \quad
    \hat{H}_{ij} =
        \left.
        \frac{\partial^2 \operatorname{cost}}{\partial b_i \partial b_j}
        \right|_{\mathbf{b}=\hat{\mathbf{b}}}
$$

The estimated covariance matrix of the estimated parameters
is approximated by the inverse of the Hessian,
which can be justified with the Laplace approximation:
{cite:p}`mackay_2005_information`.

$$
    \hat{C}_{\hat{b}_i,\hat{b}_j} = \bigl(\hat{\mathbf{H}}^{-1}\bigr)_{ij}
$$

This covariance matrix characterizes the uncertainties of the model parameters
and thus also of the autocorrelation integral.
More accurate covariance estimates can be obtained with Monte Carlo sampling,
but this is not implemented in STACIE.
Note that this covariance only accounts for the uncertainty due to noise in the spectrum,
which is acceptable if the cutoff frequency is a fixed value.
However, in STACIE, the cutoff frequency is also fitted,
meaning that the uncertainty due to the cutoff must also be accounted for.
This will be discussed in the [next section](cutoff.md).

:::{note}
The estimated covariance has no factor $N_\text{fit}/(N_\text{fit} - N_\text{par})$,
where $N_\text{fit}$ is the amount of data in the fit and $N_\text{par}$ is the number of parameters.
This factor is specific to the case of (non)linear regression with normal deviates
of which the standard deviation is not known *a priori* {cite:p}`millar_2011_maximum`.
Here, the amplitudes are Gamma-distributed with a known shape parameter.
Only the scale parameter at each frequency is predicted by the model.
:::

## Regression Cost Z-score

When a model is too simple to explain the data,
the regression cost [Z-score](https://en.wikipedia.org/wiki/Standard_score)
can be used to quantify the goodness of fit.
This is implemented in STACIE to facilitate the detection of poorly fitted models,
which can sometimes occur if the selected model cannot explain the data for any cutoff frequency.
This Z-score is defined as:

$$
    Z_\text{cost}(\mathbf{b}) = \frac{
        \operatorname{cost}(\mathbf{b}) - \mean\left[\hat{\operatorname{cost}}(\mathbf{b})\right]
    }{
        \std\left[\hat{\operatorname{cost}}(\mathbf{b})\right]
    }
$$

The mean $\mean\left[\hat{\operatorname{cost}}(\mathbf{b})\right]$
and standard deviation $\std\left[\hat{\operatorname{cost}}(\mathbf{b})\right]$
are computed as expectation values over all possible spectra sampled
from the Gamma distribution corresponding to the model parameters $\mathbf{b}$.
For both expectation values STACIE implements computationally efficient closed-form solutions.

The Z-score is easily interpretable as a goodness of fit measure.
When the model fits the data well, the Z-score has a zero mean and unit standard deviation.
When the model is too simple and underfits the data,
the Z-score is positive and quickly exceeds the standard deviation.
For example a Z-score of 2 indicates that the model cost is two standard deviations above its mean,
suggesting that the model is too simple.

The mean is derived as follows:

$$
    \mean\left[\hat{\operatorname{cost}}(\mathbf{b})\right]
    = \sum_{k\in K}
        w(f_k|f_\text{cut})
        \mean\left[-\ln p_{\gdist(\alpha_k,\theta_k)}(\hat{C}_k)\right]
$$

where the mean is computed by sampling $\hat{C}_k$ from the distribution
$\gdist(\alpha_k,\theta_k)$.
This mean is also known as the entropy of the distribution,
with a well-known closed-form solution.
Inserting this solution into the previous equation gives:

$$
    \mean\left[\hat{\operatorname{cost}}(\mathbf{b})\right]
    = \sum_{k\in K}
        w(f_k|f_\text{cut})
        \Bigl(
            \alpha_k
            - \ln\bigl(\theta_k(\mathbf{b})\bigr)
            + \ln \Gamma(\alpha_k)
            + (1 - \alpha_k)\psi(\alpha_k)
        \Bigr)
$$

where $\psi(\alpha)$ is the [digamma](https://en.wikipedia.org/wiki/Digamma_function) function.

The standard deviation is best derived by first computing the variance of the cost function:

$$
    \var\left[\hat{\operatorname{cost}}(\mathbf{b})\right]
    = \sum_{k\in K}
        w(f_k|f_\text{cut})^2
        \var\left[-\ln p_{\gdist(\alpha_k,\theta_k)}(\hat{C}_k)\right]
$$

The variance of the cost function is also defined as an expectation value
over $\hat{C}_k$ from the distribution $\gdist(\alpha_k,\theta_k)$.
This variance can also be derived analytically, but the result is not as well-known,
so we will work it out here.
The logarithm of the probability density has only two terms that depend on the random variable $\hat{C}_k$,
which are relevant for the variance:

$$
    \begin{aligned}
        &\hspace{-1em}\var\left[-\ln p_{\gdist(\alpha_k,\theta_k)}(\hat{C}_k)\right]
        \\
        &= \var\left[
            (\alpha_k - 1)\ln(\hat{C}_k)
            - \frac{\hat{C}_k}{\theta_k(\mathbf{b})}
        \right]
        \\
        &= (\alpha_k - 1)^2 \var\left[\ln(\hat{C}_k)\right]
            + \frac{1}{\theta_k^2(\mathbf{b})} \var\left[\hat{C}_k\right]
            - 2\frac{\alpha_k - 1}{\theta_k(\mathbf{b})} \cov\left[\ln(\hat{C}_k),\hat{C}_k\right]
    \end{aligned}
$$

The first two terms are well-known results,
i.e. [the variance of the log-Gamma and Gamma distributions](https://en.wikipedia.org/wiki/Gamma_distribution#Properties),
respectively.

$$
    \begin{aligned}
        \var\left[\ln(\hat{C}_k)\right] &= \psi_1(\alpha_k)
        \\
        \var\left[\hat{C}_k\right] &= \alpha_k\, \theta_k^2(\mathbf{b})
    \end{aligned}
$$

where $\psi_1(\alpha)$ is the [trigamma](https://en.wikipedia.org/wiki/Trigamma_function) function.
The only term that requires some more work is the third term:

$$
    \cov\left[\ln(\hat{C}_k),\hat{C}_k\right]
    = \mean\left[\ln(\hat{C}_k)\,\hat{C}_k\right]
      - \mean\left[\ln(\hat{C}_k)\right]\,\mean\left[\hat{C}_k\right]
$$

A [derivation of the first term](https://statproofbook.github.io/P/gam-xlogx) can be found in the
wonderful online [book of statistical proofs](https://statproofbook.github.io/).
The second term contains well-known [expectation values of the Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution#Properties).
The results are:

$$
    \begin{aligned}
        \mean\left[\ln(\hat{C}_k)\,\hat{C}_k\right]
        &= \alpha_k\,\theta_k(\mathbf{b})
           \Bigl(\psi(\alpha_k+1) + \ln\bigl(\theta_k(\mathbf{b})\bigr)\Bigr)
        \\
        \mean\left[\ln(\hat{C}_k)\right]
        &= \psi(\alpha_k) + \ln\bigl(\theta_k(\mathbf{b})\bigr)
        \\
        \mean\left[\hat{C}_k\right]
        &= \alpha_k\, \theta_k(\mathbf{b})
    \end{aligned}
$$

The covariance can now be worked out by making using of the
[well-known recurrence relation of the digamma function](https://en.wikipedia.org/wiki/Digamma_function#Recurrence_formula_and_characterization):

$$
    \begin{aligned}
        \cov\left[\ln(\hat{C}_k),\hat{C}_k\right]
        &= \alpha_k\,\theta_k(\mathbf{b})
           \bigl(\psi(\alpha_k+1) - \psi(\alpha_k)\bigr)
        \\
        &= \theta_k(\mathbf{b})
    \end{aligned}
$$

Putting it all together,
we find the variance of the logarithm of the probability density of the Gamma distribution:

$$
    \var\left[-\ln p_{\gdist(\alpha_k,\theta_k)}(\hat{C}_k)\right]
    = (\alpha_k - 1)^2 \psi_1(\alpha_k) - \alpha_k + 2
$$

The standard devation in the Z-score finally becomes:

$$
    \std\left[\hat{\operatorname{cost}}(\mathbf{b})\right]
    = \sqrt{\sum_{k\in K}
        w(f_k|f_\text{cut})^2
        \Bigl(
            (\alpha_k - 1)^2 \psi_1(\alpha_k) - \alpha_k + 2
        \Bigr)
    }
$$

It is noteworthy that the standard deviation is independent of the model parameters $\mathbf{b}$.
