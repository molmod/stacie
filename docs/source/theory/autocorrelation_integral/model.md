# Model Spectrum

## Exponential Tail Model

This model represents the autocorrelation function
as the sum of a short-term component and a (periodic) exponentially decaying tail:

$$
c_\Delta = c_\Delta^\text{short} + c_\Delta^\text{tail}
$$


### Short-term component

The short-term component is non-zero only for small positive or negative time lags:

$$
c_\Delta^\text{short} \neq 0 & \quad\text{if } |\Delta| \le \Delta_\text{short}
\\
c_\Delta^\text{short} = 0 & \quad\text{if } \Delta_\text{short} \lt |\Delta| \le N/2
$$

Because summations in DFT are commonly taken from $0$ to $N-1$, we rewrite this as:


$$
c_\Delta^\text{short}
    \neq 0 &\qquad \forall\,\Delta\in\lbrace
        0, \ldots, \Delta_\text{short},
        N-\Delta_\text{short}, \ldots, N-1
    \rbrace
\\
c_\Delta^\text{short}
    = 0 &\qquad \forall\,\Delta\in\lbrace
        \Delta_\text{short}+1, \ldots, N-\Delta_\text{short} -1
    \rbrace
$$

### Tail Component

The tail component is the periodic repetition (sum over $i$) of two exponential functions:
one decaying for positive time lags and one for negative time lags.
For $\Delta \in \lbrace 0, \ldots, N-1 \rbrace$, this can be written as:

$$
c_\Delta^\text{tail}
    = \frac{a_\text{tail}}{M} (r^\Delta + r^{N-\Delta}) \sum_{i=0}^\infty r^{iN}
$$

with $r < 1$.
The factor $\frac{1}{M}$ normalizes the tail component so that it sums to $a_\text{tail}$.

$$
M
    &= \left(\frac{1-r^N}{1-r} + r^N \frac{1-r^{-N}}{1-r^{-1}}\right) \sum_{i=0}^\infty r^{iN}
\\
    &= \left(1-r^{-N}\right) \frac{1 + r}{1-r} \sum_{i=0}^\infty r^{iN}
$$

We may absorb the infinite sum into the normalization constant to simplify the model:

$$
c_\Delta^\text{tail}
    & = \frac{a_\text{tail}}{M'} (r^\Delta + r^{N-\Delta})
\\
M'
    &= \left(1-r^{-N}\right) \frac{1 + r}{1-r}
$$


The exponential decay of the tail component is characterized by
a correlation time, $\tau_\text{tail}$:

$$
    r = \exp\left(-\frac{2}{\tau_\text{tail} / h}\right)
$$

where $h$ is the time step.

### Discrete Fourier Transform

For small values of $k$, the discrete Fourier transform of the model autocorrelation function becomes:

$$
C^\text{exp-tail}_k
    &\approx
    a_\text{short} + \frac{a_\text{tail}}{M'} \left(
        \frac{1-r^N}{1 - r \omega^{-k}} + r^N\frac{1-r^{-N}}{1 - r^{-1} \omega^{-k}}
    \right)
\\
    &\approx
    a_\text{short} + \frac{a_\text{tail}}{M'} \left(
        \frac{1-r^N}{1 - r \omega^{-k}} + \frac{1-r^N}{1 - r \omega^k} - \left(1-r^N\right)
    \right)
\\
    &\approx
    a_\text{short} + \frac{a_\text{tail}}{M'} \left(1-r^N\right) \left(
         \frac{2 - r(\omega^k + \omega^{-k})}{1 - r(\omega^k + \omega^{-k}) + r^2} - 1
    \right)
\\
    &\approx
    a_\text{short} + a_\text{tail} \frac{1-r}{1+r} \left(
         \frac{2 - r(\omega^k + \omega^{-k})}{1 - r(\omega^k + \omega^{-k}) + r^2} - 1
    \right)
$$

For the first term we assumed $\omega^{k\Delta_\text{short}}\approx 1$.
Finally, we can substitute $\omega^k + \omega^{-k} = 2\cos(2\pi k/N)$:

$$
C^\text{exp-tail}_k
    \approx
    a_\text{short} + a_\text{tail} \frac{1-r}{1+r} \left(
         2\frac{1 - r\cos(2\pi k/N)}{1 - 2r\cos(2\pi k/N) + r^2} - 1
    \right)
$$


This model can be fitted to the low-frequency part of an empirical autocorrelation function.
Once the parameters are fitted to an empirical spectrum, one finds:

$$
    \eta \approx \hat{\eta} = \hat{C}^\text{exp-tail}_0 = \hat{a}_\text{short} + \hat{a}_\text{tail}
$$

## White Noise Model

In the limit of $a_\text{tail} \rightarrow 0$ or $\tau_\text{tail} \rightarrow 0$,
the exponential tail model reduces to a white noise model:

$$
    C^\text{white}_k \approx a_\text{white}
$$

This model can be useful when the empirical spectrum has no noticeable maximum
in the zero frequency limit.
