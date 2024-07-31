# Notation

The following notation is used throughout Stacie's documentation.

## Special functions

- $\Gamma(z)$ is the Gamma function.
- $\gamma(z, x)$ is the lower incomplete Gamma function.

## Statistics

- $p_x(x)$ is the probability density function of $\hat{x}$.

- A hat is used to denote a stochastic quantity, including functions of stochastic quantities.
  This is more general than the common practice of using hats for statistical estimates only.
  We find it useful to identify all stochastic variables clearly.
  For example:

    - If $\mathcal{I}$ is the ground truth of the autocorrelation integral,
      then $\hat{\mathcal{I}}$ is an estimate of $\mathcal{I}$.
    - A sample point from a distribution $p_a(a)$ is denoted as $\hat{a}$.
    - A realization of a continuous stochastic process $p_{a(t)}[a]$ is written as $\hat{a}(t)$.
    - Similarly, a sample from a discrete stochastic process $p_{a_n}[a]$ is written as $\hat{a}_n$.

- Expectation values are denoted as:

    - $\mean[\cdot]$ is the mean operator.
    - $\var[\cdot]$ is the variance operator.
    - $\cov[\cdot,\cdot]$ is the covariance operator.

- The Gamma distribution with shape $\kappa$ and scale $\theta$ is denoted as:

    $$
        p_{\gdist(\kappa,\theta)} (x)
        = \frac{1}{\theta^\kappa \Gamma(\kappa)} x^{k - 1} e^{-x/\theta}
    $$

- The Chi-squared distribution with $\nu$ degrees of freedom is a special case of the Gamma distribution:

    $$
        p_{\chi^2_\nu} (x)
        = \frac{1}{2^{\nu/2} \Gamma(\nu/2)} x^{\nu/2 - 1} e^{-x/2}
        = p_{\gdist(\nu/2,2)} (x)
    $$


## Discrete Fourier Transform

- $x_n$ is an element of a real periodic sequence $\mathbf{x}$ with period $N$.
- $\mathbf{X} = \mathcal{F}[\mathbf{x}]$ is the discrete Fourier transform of the sequence,
  complex and periodic with period $N$.
- When $S$ samples of the sequence are considered, they are denoted as $\mathbf{x}^s$
  with elements $x^s_n$.
  Their discrete Fourier transforms are $\mathbf{X}^s$ with elements $X^s_n$.
- Hats are added if the sequences are stochastic.
