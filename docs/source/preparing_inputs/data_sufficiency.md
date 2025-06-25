# How to Prepare Sufficient Inputs for STACIE?

This section explains how to achieve a desired relative error $\epsilon_\text{rel}$
of the autocorrelation integral estimate, $\hat{\mathcal{I}}$.
The preparation of sufficient inputs consists of two steps:

1. First, we guesstimate the number of independent sequences, $M$, required
   to achieve the desired relative error.
2. Second, a test is proposed to verify that the number of steps in the input sequencess, $N$,
   is sufficient to achieve the desired relative error.
   Because this second step requires information that is not available *a priori*,
   it involves an analysis with STACIE of a preliminary set of input sequences.
   This will reveal whether the number of steps in the input sequences is sufficient.
   If not, the inputs must be extended, e.g., by running additional simulations or measurements.

## Step 1: Guesstimate the Number of Independent Sequences

Because the amplitudes of the (rescaled) sampling PSD are Gamma-distributed,
one can show that the relative error of the PSD (mean divided by the standard deviation)
is given by:

$$
     \frac{\std[\hat{I}_k]}{\mean[\hat{I}_k]} = \sqrt{\frac{2}{\nu_k}}
$$

where $\nu_k$ is the number of degrees of freedom of the sampling PSD at frequency $k$.
For most frequencies, we have $\nu_k=2M$.
(See [Parameter Estimation](../theory/statistics.md) for details.)
Because we are only interested in an coarse estimate of the required number of independent sequences,
we will use $\nu_k=2M$ for all frequencies.

Let us assume for simplicity that we want to fit a white noise spectrum,
which can be modeled with a single parameter, namely the amplitude of the spectrum.
In this case, this single parameter is also the autocorrelation integral.
By taking the average of the PSD over the first $N_\text{eff}$ frequencies,
the relative error of the autocorrelation integral is approximately given by:

$$
     \epsilon_\text{rel} = \frac{1}{\sqrt{M N_\text{eff}}}
$$

In general, for any model, we recommend fitting to at least $N_\text{eff}=20\,P$ points.
Substituting this good practice into the equation above,
we find the following estimate of the number of independent sequences $M$:

$$
     M \approx \frac{1}{20\,P\,\epsilon_\text{rel}^2}
$$

Given the simplicity and the drastic assumptions made,
this is only a guideline and should not be seen as a strict rule.

From our practical experience, $M=10$ is a low number and $M=500$ is quite high.
For $M<10$, the results are often rather poor and possibly a bit confusing.
In this low-data regime, the sampling PSD is extremely noisy.
While we have validated STACIE in this low-data regime with the ACID test set,
the visualization of the spectrum will not be very informative for low $M$.

A single molecular dynamics simulation often provides more than one independent sequence.
The following table lists $M$ (for a single simulation) for the transport properties discussed
in the [Properties](../properties/index.md) section.

| Transport Property |  $M$  |
| ------------------ | :---: |
| Bulk Viscosity | $1$ |
| Thermal Conductivity | $3$ |
| Ionic Electrical Conductivity | $3$ |
| Shear Viscosity | $5$ |
| Diffusivity | $3N_\text{atom}$ |

This means that in most cases (except for diffusivity), multiple independent simulations
are required to achieve a good estimate of the transport property.
While diffusivity may seem to be a very forgiving case,
it is important to note that displacements of particles in a liquid are often highly correlated.
STACIE assumes its inputs to be independent,
which is not the case for particle velocities when studying self-diffusivity in a liquid.
A correct treatment of uncertainty quantification in this case is a topic of ongoing research.

## Step 2: Test the Sufficiency of the Number of Steps and Increase if Necessary

There is no simple way to know *a priori* the required number of steps in the input sequences.
Hence, we recommend first generating inputs with about $400\,P$ steps,
where $P$ is the number of model parameters, and analyzing these inputs with STACIE.
With this choice, the first $20 P$ points that are ideally used for fitting
will be a factor $10$ below the Nyquist frequency,
which is a minimal first attempt to identify the low-frequency part of the spectrum.
Using these data as inputs, you will obtain a first estimate
of the autocorrelation integral and its relative error.
If the relative error is larger than the desired value,
you can extend the input sequences with additional steps and repeat the analysis.

Note that for some applications, $400\,P$ steps may be far too short,
meaning that you will need to extend your inputs a few times
before you get a clear picture of the relative error.
It is not uncommon to run into problems with storage quota in this scenario.
To reduce the storage requirements, [block averages](block_averages.md) can be helpful.

In addition to the relative error, there are other indicators to monitor
the quality of the results.
If any of the following criteria are not met,
we recommend extending the input sequences with additional steps
and repeating the analysis with STACIE:

- The effective number of points used in the fit, which is determined by the cutoff frequency,
  should be larger than 20 times the number of model parameters.
- The Z-score computed for the regression cost and the cutoff criterion
  should be smaller than 2.
  Note that the Z-scores may also be large for other reasons than insufficient data.
  This may also occur when the functional form of the model can never match the data,
  e.g. fitting a white noise model to a spectrum that has a non-zero slope.
- When using the Pade model, the total simulation time should be sufficient
  to resolve the zero-frequency peak of the spectrum.
  The width of the peak can be derived from
  [the Pade model](../theory/model.md)
  and is $1/2\pi\tau_\text{exp}$,
  where $\tau_\text{exp}$ is the exponential correlation time.
  Because the resolution of the frequency axis of the power spectrum is $1/T$,
  where $T$ is the total simulation time,
  ample frequency grid points in this first peak are guaranteed when:

  $$
        T \gg 2\pi\tau_\text{exp}
  $$

  For example, $T \approx 20\pi\tau_\text{exp}$ will provide a decent resolution.
  When using a discrete time step $h$, the corresponding number of steps is:

  $$
        N \approx 20\pi\tau_\text{exp}/h
  $$

  When STACIE estimates a large exponential correlation time, e.g. $\tau_\text{exp} > T/(20 \pi)$,
  it has derived this value from a very sharp spectral peak at zero frequency.
  In this case, the peak width is artificially broadened due to
  [spectral leakage](https://en.wikipedia.org/wiki/Spectral_leakage),
  which results in an underestimation of the correlation time.
  Hence, the true exponential correlation time is then even larger than the estimated value.

Finally, it is recommended that you use sequences whose length is a power of two,
or at least a product of small prime numbers.
NumPy's FFT algorithm used in STACIE is optimized for such sequences
and becomes significantly slower for sequence lengths with large prime factors.
A good strategy for adhering to this recommendation is to start with a sequence length
equal to the first power of two greater than $400 P$, where $P$ is the number of model parameters.
Then repeatedly double the sequence length
until the analysis with STACIE indicates that the number of steps is sufficient.
This approach also facilitates increasing the block size by factors of 2 *a posteriori*,
when working with [block averages](block_averages.md) to reduce storage requirements.
