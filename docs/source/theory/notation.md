# Notation

The following notation is used throughout STACIE's documentation.

## Special functions

- $\Gamma(z)$ is the Gamma function.
- $\gamma(z, x)$ is the lower incomplete Gamma function.

## Statistics

- Several symbols are used to denote time:

    - $t$ is an absolute time.
    - $t_0$ is a reference point on the time axis.
    - $\Delta_t$ is a time difference or lag.
    - $\tau$ denotes a relaxation or autocorrelation time,
      and usually has a subscript $\text{int}$ or $\text{exp}$
      to distinguish between integrated and exponential autocorrelation times.
    - $h$ is the time step of a discretized time axis (with equal spacing between the grid points).
    - Integer steps on a discretized time axis are denoted by
      indices $n$ or $m$; the difference between them is $\Delta$.

- $p_x(x)$ is the probability density function of $\hat{x}$.

- A hat is used for all stochastic quantities, including functions of stochastic quantities.
  This is more general than the common practice of using hats for statistical estimates only.
  We find it useful to clearly identify all stochastic variables.
  For example:

    - If $\mathcal{I}$ is the ground truth of the autocorrelation integral,
      then $\hat{\mathcal{I}}$ is an estimate of $\mathcal{I}$.
    - The sampling variance is denoted as $\hat{\sigma}^2_{\mathcal{I}}$.
    - The sampling covariance is denoted as $\hat{C}_{a,b}$.
    - The sampling covariance matrix of two stochastic vectors is denoted as
      $\hat{\mathbf{C}}_{\mathbf{a},\mathbf{b}}$.
    - A sample point from a distribution $p_a(a)$ is denoted as $\hat{a}$.
    - A realization of a continuous stochastic process $p_{a(t)}[a]$ is written as $\hat{a}(t)$.
    - Similarly, a sample from a discrete stochastic process $p_{a_n}[a]$ is written as $\hat{a}_n$.

- Expected values are denoted as:

    - $\mean[\cdot]$ is the mean operator.
    - $\var[\cdot]$ is the variance operator.
    - $\std[\cdot]$ is the standard deviation operator.
    - $\cov[\cdot,\cdot]$ is the covariance operator.

- The [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)
  with shape $\alpha$ and scale $\theta$ is denoted as:

    $$
        p_{\gdist(\alpha,\theta)} (x)
        = \frac{1}{\theta^\alpha \Gamma(\alpha)} x^{\alpha - 1} e^{-x/\theta}
    $$

- The [Chi-squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution)
  with $\nu$ degrees of freedom is a special case of the Gamma distribution:

    $$
        p_{\chi^2_\nu} (x)
        = \frac{1}{2^{\nu/2} \Gamma(\nu/2)} x^{\nu/2 - 1} e^{-x/2}
        = p_{\gdist(\nu/2,2)} (x)
    $$

## Discrete Fourier Transform

- $x_n$ is an element of a real periodic sequence $\mathbf{x}$ with period $N$.
- $\mathbf{X} = \mathcal{F}[\mathbf{x}]$ is the discrete Fourier transform of the sequence,
  complex and periodic with period $N$.
- When $M$ samples of the sequence are considered, they are denoted as $\mathbf{x}^{(m)}$
  with elements $x^{(m)}_n$.
  Their discrete Fourier transforms are $\mathbf{X}^{(m)}$ with elements $X^{(m)}_k$.
- The grid spacing on the frequency axis is $1/hN$, where $h$ is the spacing of the time axis.
- Frequency grid points are labeled by an index $k$, such that the $k$th frequency is $k/hN$.
- Hats are added if the sequences are stochastic.
