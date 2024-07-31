# Parameter Estimation

## Statistics of the Power Spectrum

Consider a periodic random real sequence $\hat{\mathbf{x}}$ with elements $\hat{x}_n$ and period $N$.
For practical purposes it is sufficient to consider one period of this infinitely long sequence.
The mean of the sequence is zero and its covariance is $\cov[\hat{x}_n \,,\, \hat{x}_m]$.

The distribution of sequences is stationary,
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
As a result, the covariance of the sequence $\hat{\mathbf{X}}$ becomes diagonal.
To make this derivation self-contained, we write out the mean and covariance of $\hat{X}_k$ explicitly.
Note that the operators $\mean[\cdot]$, $\var[\cdot]$ and $\cov[\cdot,\cdot]$
are expectation values over all possible realizations of the sequence.

For the expectation value of the Fourier transform,
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
            \frac{1}{N} \sum_{m=0}^{N-1} \omega^{km}\sum_{n=0}^{N-1} \hat{x}_{n} \omega^{-kn}
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

Finally, the covariance of the Fourier transform of the input sequence takes the following form:

$$
    \cov[\hat{X}^*_k \,,\, \hat{X}_\ell] = \delta_{k,\ell} \mean\Bigl[|\hat{X}_k|^2\Bigr] = N \delta_{k,\ell} C_k
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
This simple statistical structure makes the spectrum an easy starting point for further analysis.
In comparison, the autocorrelation function itself has non-trivial correlated uncertainties,
{cite:p}`bartlett_1955_introduction,boshnakov_1996_bartlett,francq_2009_bartlett`
making it difficult to fit models directly to it (or its running integral).

If we further assume that the sequence $\hat{\mathbf{x}}$ is the result of a periodic Gaussian process,
the Fourier transform is normally distributed.
In this case, the empirical power spectrum follows a scaled Chi-squared distribution
{cite:p}`ercole_2017_accurate`.
For notational consistency, we will use the Gamma distribution:

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

Spectra are often computed by averaging them over $S$ samples to reduce the variance.
In this case, the $S$-averaged empirical spectrum is distributed as:

$$
    \hat{C}_0=\frac{1}{NS}\sum_{s=1}^S|\hat{X}^s_0|^2
    &\sim \gdist(\textstyle\frac{S}{2},\textstyle\frac{2}{S}C_0)
    \\
    \hat{C}_{N/2}=\frac{1}{NS}\sum_{s=1}^S|\hat{X}^s_{N/2}|^2
    &\sim \gdist(\textstyle\frac{S}{2},\textstyle\frac{2}{S}C_{N/2})
    \quad \text{if $N$ is even}
    \\
    \hat{C}_k=\frac{1}{NS}\sum_{s=1}^S|\hat{X}^s_k|^2
    &\sim \gdist(S,\textstyle\frac{1}{S}C_0)
    \quad \text{for } 0<k<N \text { and } k \neq N/2
$$

(lmax-target)=
## Likelihood Maximization

To facilitate the treatment of the (log) likelihood,
we write the empirical spectrum as $\hat{C}_k$
and introduce $\nu_k$ for the number of degrees of freedom contributing to each component.
($S$ if there is only a real component, $2S$ if there are both real and imaginary ones.)

The model is not fitted to the entire spectrum, but only to a subset of the data,
i.e., all indexes $k \in K$, where the set $K$ is fixed before maximizing the likelihood.
There are two reasons to exclude parts of the spectrum from the fit:

- Frequency 0 corresponds to the DC component and may contain an unknown contribution.
  For example, this is the case when the input sequences $\hat{\mathbf{x}}$
  have a non-zero mean that cannot be easily subtracted.
  (Subtracting the sample mean typically produces a biased DC component
  and is therefore not recommended.)
- The Exponential Tail model of Stacie is only applicable to low-frequency data.
  It is not designed to describe all features in a spectrum.

The log likelihood of the model $C^\text{model}_k(\mathbf{b})$ with parameter vector $\mathbf{b}$ becomes:

$$
    \ln\mathcal{L}(\mathbf{b})
    &=\sum_{k\in K} \ln p_{\gdist(\kappa_k,\theta_k)}(\hat{C}_k)
    \\
    &=\sum_{k\in K}
      -\ln \Gamma(\kappa_k)
      - \ln\bigl(\theta_k(\mathbf{b})\bigr)
      + (\kappa_k - 1)\ln\left(\frac{\hat{C}_k}{\theta_k(\mathbf{b})}\right)
      - \frac{\hat{C}_k}{\theta_k(\mathbf{b})}
$$

with

$$
    \kappa_k &= \frac{\nu_k}{2}
    \\
    \theta_k(\mathbf{b}) &= \frac{2 C^\text{model}_k(\mathbf{b})}{\nu_k}
$$

This log-likelihood is maximized to estimate the model parameters.
The zero-frequency limit of the fitted model is then the estimate of the autocorrelation integral.

For compatibility with the SciPy optimizers,
the cost function $\ell(\mathbf{b}) = -\ln \mathcal{L}(\mathbf{b})$ is minimized.
Stacie implements first and second derivatives of $\ell(\mathbf{b})$,
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
is approximated by the inverse of the Hessian:
{cite:p}`millar_2011_maximum`.

$$
    \widehat{\cov}[\hat{b}_i,\hat{b}_j] = (-\hat{\mathbf{H}}^{-1})_{ij}
$$

This covariance matrix is used to estimate the uncertainties on the model parameters
and thus also on the autocorrelation integral.
More accurate covariance estimates can be obtained with Monte Carlo sampling,
but this is not implemented in Stacie.

:::{note}
The estimated covariance has no factor $N_\text{fit}/(N_\text{fit} - N_\text{par})$,
where $N_\text{fit}$ is the amount of data in the fit
and $N_\text{par}$ is the number of parameters.
This is factor is specific for the case of (non)linear regression with normal deviates of
which the standard deviation is not known a priori {cite:p}`millar_2011_maximum`.
Here, the amplitudes are Gamma-distributed with a known shape parameter.
Only the scale parameter at each frequency is predicted by the model.
:::
