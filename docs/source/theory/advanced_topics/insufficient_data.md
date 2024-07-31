# How to Detect and Address Insufficient Data

If the spectrum analyzed by Stacie contains too little information,
i.e., there is not enough independent data in the input sequences,
the estimated parameters and uncertainties will be unreliable.
The most common consequence is that the autocorrelation integral will be underestimated,
and that the error bars will be small compared to this bias.

There are several heuristics that can be used to detect a lack of data.
Ideally, all of the following conditions are met to obtain robust results.
If not, one should be cautious in interpretating the results and consider generating more data.

- The individual the input sequences must be (much) longer than the exponential correlation time, $\tau_\text{exp}$.

    > The slowest time correlations can only be detected
    > if these slow changes in the inputs are repeated a few times.
    > This is a rather general requirement, not specific to Stacie.
    > If this condition is not met, the relevant information for estimating
    > the autocorrelation integral is simply missing.
    > More details can be found in the section on the
    > [autocorrelation time](../properties/autocorrelation_time.md).

- The product $2\pi\tau_\text{exp} f_\text{cut}$,
  where $f_\text{cut}$ is the cutoff frequency
  used to determine the number of fitting points,
  should be of the order of 1.

    > - If this product is much less than 1,
    >   the selected part of the spectrum does not have enough relevant features
    >   to fit the Exponential Tail Model.
    >   The selected part of the spectrum should show some trend,
    >   except for the trivial case where the inputs are white noise.
    > - When this product is much greater than 1,
    >   only a small part of the fitting data will be related to
    >   the zero-frequency limit of the spectrum,
    >   which may result in a poor estimate of the autocorrelation integral.

- The relative error of the autocorrelation integral should be less than 0.1.

    > Large relative errors indicate that the exponential tail model is being fitted to noisy
    > spectral data with few data points.
    > In general, error estimates become less reliable when they are relatively large.

There are two ways to generate more data:

1. You can extend the input sequences with additional time steps.
   This is useful when the resolution of the frequency axis is low,
   for example, when you find that the exponential tail model is fitted
   to a small number of data points.
2. You can expand the number of input sequences and keep their length fixed.
   This is mainly useful when the frequency resolution seems sufficient
   but a lower variance on the spectrum is desired.
