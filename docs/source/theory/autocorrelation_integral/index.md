# Autocorrelation Integral

This theory section deals only with the autocorrelation integral as such.
The (physical) [properties](../properties/index.md) associated with this integral
are discussed in the next section.

Some of the derivations in the theory section can also be found in other references.
They are included here to make the theory more accessible
and to explain all the details needed to implement STACIE.

First, the [notation](notation.md) is fixed,
and an [overview](overview.md) is given of how STACIE works.
The three main parts of the derivation consist of:

- A [model](model.md) for the low-frequency part of the power spectrum,
- the algorithm to [estimate the parameters](statistics.md) in this model,
  from which the autocorrelation integral and its uncertainty can be derived,
- and the algorithm to determine the [frequency cutoff](cutoff.md) used
  to identify the low-frequency part of the spectrum.

```{toctree}
:hidden:

notation.md
overview.md
model.md
statistics.md
cutoff.md
```
