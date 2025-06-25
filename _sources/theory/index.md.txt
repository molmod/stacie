# Theory

This section focuses solely on the autocorrelation integral itself.
The (physical) [properties](../properties/index.md) associated with this integral
are discussed later.

Some derivations presented here can also be found in other sources.
They are included to enhance accessibility
and to provide all the necessary details for implementing STACIE.

First, the [notation](notation.md) is defined,
and an [overview](overview.md) is presented of how STACIE works.
The derivation comprises three main parts:

- A [model](model.md) for the low-frequency part of the power spectrum,
- an algorithm to [estimate the parameters](statistics.md) of this model,
  from which the autocorrelation integral and its {term}`uncertainty` can be derived,
- and an algorithm to determine the [frequency cutoff](cutoff.md) used
  to identify the low-frequency part of the spectrum.

```{toctree}
:hidden:

notation.md
overview.md
model.md
statistics.md
cutoff.md
```
