# How to Prepare Sufficient Inputs for STACIE?

This section explains how to achieve a desired relative error $\epsilon_\text{rel}$
of the autocorrelation integral estimate, $\hat{\mathcal{I}}$.
The preparation of sufficient inputs consists of two steps:

1. First, we guesstimate the number of independent sequences required
   to achieve the desired relative error.
2. Second, a test is proposed to verify that the number of steps in the input sequences
   is sufficient to achieve the desired relative error.
   Because this second step requires information that is not available *a priori*,
   it involves an analysis with STACIE of a preliminary set of input sequences.
   This will reveal whether the number of steps in the input sequences is sufficient.
   If not, the inputs must be extended, e.g., by running additional simulations or measurements.

## 1. Guesstimating the Number of Independent Sequences

Because the amplitudes of the (rescaled) sampling PSD are Gamma-distributed,
one can show that the relative error of the PSD (mean divided by the standard deviation)
is given by:

$$
     \frac{\std[\hat{I}_k]}{\mean[\hat{I}_k]} = \sqrt{\frac{2}{\nu_k}}
$$

where $\nu_k$ is the number of degrees of freedom of the sampling PSD at frequency $k$.
For most frequencies, we have $\nu_k=2M$.
(See [Parameter Estimation](../autocorrelation_integral/statistics.md) for details.)
Because we are only interested in an coarse estimate of the required number of independent sequences,
we will use $\nu_k=2M$ for all frequencies.

Let us assume for simplicity that we want to fit a white noise spectrum,
which can be modeled with a single parameter, namely the amplitude of the spectrum.
In this case, this single parameter is also the autocorrelation integral.
By taking the average of the PSD over the first $N$ frequencies,
the relative error of the autocorrelation integral is approximately given by:

$$
     \epsilon_\text{rel} = \frac{1}{\sqrt{M N}}
$$

In general, for any model, we recommend fitting to at least $N=20\,P$ points.
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
the visualization of the spectrum is not very informative.

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

## Testing the Sufficiency of the Number of Steps

There is no simple way to know *a priori* the required number of steps in the input sequences.
Hence, we recommend first generating inputs with about $400\,P$ steps,
where $P$ is the number of model parameters, and analyzing these inputs with STACIE.
This will provide a first estimate of the autocorrelation integral and its relative error.
If the relative error is larger than the desired value,
you can extend the input sequences with additional steps and repeat the analysis.

Note that for some applications, $400\,P$ steps may be far too short,
meaning that you will need to extend your inputs a few times
before you get a clear picture of the relative error.
It is not uncommon to run into problems with storage quota in this scenario.
To reduce the storage requirements, [block averages](block_averages.md) can be helpful.

In addition to the relative error, there are other indicators to monitor
the quality of the results:

- The effective number of points used in the fit, which is determined by the cutoff frequency,
  should be larger than 20 times the number of model parameters.
- When using the Pade model, the total simulation time should be sufficient
  to resolve the zero-frequency peak of the spectrum.
  The width of the peak can be derived from
  [the Pade model](../autocorrelation_integral/model.md)
  and is $1/2\pi\tau_\text{exp}$.
  Because the resolution of the frequency axis of the power spectrum is $1/T$,
  ample frequency grid points in this first peak are guaranteed when:

  $$
        T \gg 2\pi\tau_\text{exp}
  $$

  For example, $T = 20\pi\tau_\text{exp}$ will provide a decent resolution.
