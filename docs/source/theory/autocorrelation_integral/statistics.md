# Parameter Estimation

Before we discuss how to fit a model to spectral data,
we first review the statistics of the sampling {term}`PSD`.
Given these statistical properties,
we can derive the likelihood that some model parameters explain the observed PSD.

## Statistics of the Sampling Power Spectral Distribution

When constructing an estimate of a discrete PSD from a finite mount if data,
it is bound to contain some uncertainty, which will be characterized below.
The estimate of the PSD is sometimes also called
the [periodogram](https://en.wikipedia.org/wiki/Periodogram) or the (empirical) power spectrum.

Consider a periodic random real sequence $\hat{\mathbf{x}}$ with elements $\hat{x}_n$ and period $N$.
For practical purposes it is sufficient to consider one period of this infinitely long sequence.
The mean of the sequence is zero and its covariance is $\cov[\hat{x}_n \,,\, \hat{x}_m]$.
The distribution of sequence is stationary,
i.e. each time translation of a sequence results in an equally probable sample.
As a result, the covariance has a circulant structure:

$$
    \cov[\hat{x}_n \,,\, \hat{x}_m] = c_\Delta = c_{-\Delta}
$$

with $\Delta=n-m$.
Thus, we can express the covariance with a single index and treat it as a real periodic sequence,
albeit not stochastic.
$c_\Delta$ is also known as the autocovariance or autocorrelation function
of the stochastic process,
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
Note that the operators $\mean[\cdot]$, $\var[\cdot]$ and $\cov[\cdot,\cdot]$
are expected values over all possible realizations of the sequence.

For the expected value of the Fourier transform,
we take advantage of the fact that all time translations of $\hat{\mathbf{x}}$
belong to the same distribution.
We can explicitly compute the average over all time translations,
in addition to computing the mean, without loss of generality.
In the last steps, the index $n$ is relabeled to $n-m$ and some factors are rearranged,
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
Halfway, the summation index $n$ is written as $n=\Delta+m$.

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
In comparison, the ACF has non-trivial correlated uncertainties,
{cite:p}`bartlett_1980_introduction,boshnakov_1996_bartlett,francq_2009_bartlett`
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

Note that $\hat{X}_0$ and $\hat{X}_{N/2}$ have only a real component,
because the input sequence $\hat{\mathbf{x}}$ is real,
which corresponds to a Chi-squared distribution with one degree of freedom.
For all other frequencies, $\hat{X}_k$ have a real and imaginary component,
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
(This is should be fine for most applications.)
This value can be set with the `switch_exponent` argument
of the [estimate_acint()](#stacie.estimate.estimate_acint) function.
We will derive how to fit parameters for a given frequency cut-off $f_\text{cut}$.
The [next section](cutoff.md) describes how to find suitable cutoffs.

To fit the model, we use a form of local regression,
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
$\mathcal{L}(\mathbf{b})$, depends on cutoff frequency.
As a result, one cannot compare the likelihood of two different cutoffs.
This is of little concern when fitting parameters for a fixed cutoff,
but it is important to keep in mind when searching for suitable cutoffs.
:::

For compatibility with the SciPy optimizers,
the cost function $\ell(\mathbf{b}) = -\ln \mathcal{L}(\mathbf{b})$ is minimized.
STACIE implements first and second derivatives of $\ell(\mathbf{b})$,
and also a good initial guess of the parameters, using efficient vectorzed NumPy code.
These features make the optimization of the parameters both efficient and reliable.

The Hessian computed with the estimated parameters, $\ell(\hat{\mathbf{b}})$,
must be positive definite.
(If non-positive eigenvalues are found, the optimization is treated as failed.)

$$
    \hat{\mathbf{H}} > 0 \quad \text{with}
    \quad
    \hat{H}_{ij} =
        \left.
        \frac{\partial^2 \ell}{\partial b_i \partial b_j}
        \right|_{\mathbf{b}=\hat{\mathbf{b}}}
$$

The estimated covariance matrix of the estimated parameters
is approximated by the inverse of the Hessian, which can be justified with the Laplace approximation:
{cite:p}`mackay_2005_information`.

$$
    \hat{C}_{\hat{b}_i,\hat{b}_j} = \bigl(\hat{\mathbf{H}}^{-1}\bigr)_{ij}
$$

This covariance matrix characterizes the uncertainties of the model parameters
and thus also of the autocorrelation integral.
More accurate covariance estimates can be obtained with Monte Carlo sampling,
but this is not implemented in STACIE.
Note that this covariance only accounts for the uncertainty due to noisy in the spectrum,
which is acceptable if the cutoff frequency is a fixed value.
However, in STACIE, the cutoff frequency is also fitted,
meaning that also the uncertainty due to the cutoff must be accounted for.
This will be discussed in the [next section](cutoff.md).

:::{note}
The estimated covariance has no factor $N_\text{fit}/(N_\text{fit} - N_\text{par})$,
where $N_\text{fit}$ is the amount of data in the fit
and $N_\text{par}$ is the number of parameters.
This is factor is specific for the case of (non)linear regression with normal deviates of
which the standard deviation is not known *a priori* {cite:p}`millar_2011_maximum`.
Here, the amplitudes are Gamma-distributed with a known shape parameter.
Only the scale parameter at each frequency is predicted by the model.
:::
