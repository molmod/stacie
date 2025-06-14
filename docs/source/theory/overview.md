# STACIE Algorithm Overview

The goal of STACIE is to estimate the integral of the {term}`ACF`
of a physical, continuous, time-dependent function with an infinite domain.
In practice, due to inherently finite computational resources, however,
we resort to discrete and finite time-dependent sequences.
We first formulate STACIE's goal in the continuous case
and then reformulate it in the discrete case.

## Continuous time, infinite domain

Consider an observation, $\hat{x}(t)$, of a time-dependent stochastic process.
The integral of the ACF is defined as:

$$
\mathcal{I} =
    \frac{F}{2} \int_{-\infty}^{\infty}
    c(\Delta_t) \, \mathrm{d}\Delta_t
$$

with

$$
  c(\Delta_t) = \cov \bigl[ \hat{x}(t) \,,\,\hat{x}(t + \Delta_t) \bigr]
$$

A prefactor $F$ is usually present, containing factors
such as the temperature and/or the cell volume in Green--Kubo formalisms
{cite:p}`green_1952_markoff, green_1954_markoff, kubo_1957_statistical`.
The integrand is the ACF, $c(\Delta_t)$, of the time-dependent input $\hat{x}(t)$.
It is common to integrate only from $0$ to $\infty$,
but we prefer to use the full range and compensate with the factor $\frac{1}{2}$ in front of the integral.
(The integrand is an even function of $\Delta_t$.)
The expected value is obtained by averaging over all times $t$
and all observations $\hat{x}(t)$.

Let $C(f)$ be the Fourier transform of the ACF,
which is also known as the {term}`PSD`:

$$
C(f)=\mathcal{F}[c](f)=\int_{-\infty}^\infty c(\Delta_t) e^{-i2\pi f \Delta_t} \mathrm{d} \Delta_t
$$

Then $\mathcal{I}$ is simply proportional to the DC component of the PSD,
i.e., the zero-frequency limit of the Fourier transform of the ACF:

$$
\mathcal{I} = \frac{F\, C(0)}{2}
$$

At first glance, this result seems trivial,
with no added value over the original form of the integral.
For numerical applications, this is actually a useful identity:
the sampling ACF is practically computed using the sampling PSD as an intermediate step.
When $\mathcal{I}$ is derived from the PSD,
the inverse transform to derive the ACF from the PSD can be skipped.
As we will see later, there are other advantages to using this zero-frequency limit to compute the integral.

:::{note}
Some derivations of Green--Kubo relations of transport properties,
conventionally formulated as integrals of autocorrelation functions,
also express them as the zero-frequency limit of an appropriate spectrum
{cite:p}`hansen_2013_theory`.
:::

One can always rewrite the autocorrelation integral
as a so-called Einstein--Helfand relation {cite:p}`helfand_1960_transport`, i.e.,
as the limit of the time derivative of the mean-square displacement {cite:p}`hansen_2013_theory`:

$$
    \mathcal{I} =
        F \frac{1}{2} \lim_{\Delta_t \rightarrow \infty} \frac{\mathrm{d}}{\mathrm{d}\Delta_t}
        \Bigl\langle
            \bigl|\hat{y}(t_0 + \Delta_t) - \hat{y}(t_0)\bigr|^2
        \Bigr\rangle
$$

where $\hat{y}$ is the antiderivative of $\hat{x}$:

$$
    \hat{x} = \frac{\mathrm{d}\hat{y}}{\mathrm{d}t}
$$

STACIE can also be used to evaluate such limits
by using samples of the time derivatives of $y$
as input to the computation of the PSD.

## Discretized time, periodic sequences

For simplicity, we first discuss the basic identities
in terms of the ensemble average of the discrete ACF,
which has no statistical uncertainties.
Further below, we comment on how to deal with the uncertainties,
and refer to the following sections for the details.

### In terms of ensemble averages

In analogy with the continuous infinite-time case,
the autocorrelation integral can be expressed in terms of discrete and periodic sequences, $\hat{x}_n$.
For example, such a sequence is obtained by discretizing the time axis
with a time step $h$ and a time origin $t_0$:

$$
\hat{x}_n = \hat{x}(t_0 + nh) \quad \forall\, n=0 \ldots N-1
$$

The underlying continuous function $\hat{x}(t)$, and thus $\hat{x}_n$, are not necessarily periodic.
However, because we intend to use the discrete Fourier transform and rely on its well-known properties,
we will assume in the derivations that $\hat{x}_n$ is periodic with period $N$.
In practice, this assumption has negligible effects and is only noticeable at higher frequencies,
far away from the zero-frequency limit of interest.

Due to the discretization in time,
the autocorrelation integral must be approximated with a simple quadrature rule:

$$
\mathcal{I} = F h \frac{1}{2} \sum_{\Delta=0}^{N-1} \cov \bigl[ \hat{x}_n \,,\, \hat{x}_{n+\Delta} \bigr]
$$

The summand is the discrete ACF, $c_\Delta$.
The covariance is an expected value over all $n$ and all possible realizations of the input sequence.

Let $C_k$ be the discrete Fourier transform of the autocorrelation function:

$$
C_k = \sum_{\Delta=0}^{N-1} c_\Delta \omega^{-k\Delta}
$$

with $\omega = \exp(i 2\pi/N)$.
According to (the discrete version of) the [Wiener--Khinchin theorem](statistics.md) {cite:p}`oppenheim_1999_power`,
this Fourier transform can be written in terms of the discrete PSD:

$$
C_k = \frac{1}{N}\mean \left[\left|\hat{X}_k\right|^2\right]
$$

with

$$
\hat{X}_k = \sum_{n=0}^{N-1} (\hat{x}_n - \mean[\hat{x}_n]) \omega^{-kn}
$$

In STACIE, we always work with a rescaled version of the PSD,
including the factor $Fh / 2$:

$$
  I_k = \frac{Fh}{2} C_k
$$

In this notation, the autocorrelation integral is simply
the zero-frequency limit of the PSD: $\mathcal{I} = I_0$.

### In terms of sampling estimates

So far, we have worked with ensemble averages to define the discrete ACF and PSD.
In practice, however, we must work with sampling estimates of these quantities.
To keep the notation simple, we will assume that $\mean[\hat{x}_n]=0$.
Furthermore, we will assume that we can use $M$ independent sequences of length $N$:

$$
    \hat{x}_n^{(m)} \quad \forall\, n=0 \ldots N-1 \quad \forall\, m=1 \ldots M
$$

In this case, the discrete sampling ACF is estimated as:

$$
  c_\Delta \approx \hat{c}_\Delta
    = \frac{1}{N M} \sum_{m=1}^M \sum_{n=0}^{N-1}
      \hat{x}_n^{(m)} \hat{x}_{n+\Delta}^{(m)}
$$

The discrete sampling PSD, rescaled with STACIE's conventions, becomes:

$$
  I_k \approx \hat{I}_k
    = \frac{Fh}{2N} \sum_{m=1}^M
      \left|\hat{X}_k^{(m)}\right|^2
$$

where $X_k^{(m)}$ is the discrete Fourier transform of the $m$-th sequence:

$$
  \hat{X}_k^{(m)} = \sum_{n=0}^{N-1} \hat{x}_n^{(m)} \omega^{-kn}
$$

Plotting the low-frequency part of $\hat{I}_k$
will already give a quick visual estimate of $\mathcal{I}$ with the appropriate units.

A direct computation of $\hat{I}_0$ for a limited number of input sequences
would at best yield a high-variance estimate of $\mathcal{I}$.
(This is only possible if $\mean[\hat{x}_n]=0$, which is not always the case.)

To reduce the variance of the estimate of $\mathcal{I}$, STACIE derives the zero-frequency limit
by fitting a model to the low-frequency part of the power spectrum $\hat{I}_k$.
The following theory sections explain how this estimate can be made robustly.
In summary, STACIE introduces a few [models](model.md) for the low-frequency spectrum.
The parameters in such a model are estimated with likelihood maximization,
and the parameter covariance is estimated with the Laplace approximation {cite:p}`mackay_2005_information`.
To write out the likelihood function,
the [statistical distribution](statistics.md) of the sampling PSD amplitudes must be derived.
Finally, STACIE determines up to which [cutoff](cutoff.md) frequency the model will be fitted.
For cutoffs that are too high,
the model becomes too simple to describe all the features in the spectrum,
which leads to significant underfitting.
When the cutoff is too low,
too few data points are included to obtain a low-variance estimate of $\mathcal{I}$.
This is solved by considering a grid of cutoff frequencies
and assigning weights to each grid point based on the bias-variance trade-off of the regression.
The final parameters are obtained by averaging over the grid of cutoff frequencies.
