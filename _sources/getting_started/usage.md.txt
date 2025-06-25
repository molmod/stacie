# Usage Overview

This section provides an overview of how to use STACIE.
More detailed information can be found in the remaining sections of the documentation.

The STACIE algorithm provides robust and reliable estimates of autocorrelation integrals
without requiring extensive adjustment of its settings.
Users simply provide the relevant inputs to STACIE:
the time-correlated sequences in the form of a NumPy array,
a few physical parameters (such as the time step),
and a model to fit to the spectrum.
This can be done in a Jupyter notebook for interactive work or in a Python script.

The most important inputs for STACIE are time series data on an equidistant time grid.
You can provide multiple independent sequences of the same length to reduce uncertainties.
The analysis returns a `result` object including the following attributes:

- `acint`: The integral of the autocorrelation function.
- `corrtime_int`: The *integrated* autocorrelation time.
- `corrtime_exp`: The *exponential* autocorrelation time (if supported by the selected model).

The estimated uncertainties are accessible through the
`acint_std`, `corrtime_int_std`, and `corrtime_exp_std` attributes, respectively.
In addition, intermediate results of the analysis can be accessed,
e.g., to create plots using the built-in plotting functions.

Many (transport) properties are defined in terms of an autocorrelation integral.
They require slightly different settings and preprocessing of the input data.
STACIE's documentation contains instructions for
[the properties we have tested](../properties/index.md).
In addition, we provide [worked examples](../examples/index.md)
that show in detail how STACIE is used in practice.

If you plan to produce publication-quality research with STACIE,
the analysis inevitably becomes an iterative process.
The main difficulty is providing sufficient data for the analysis,
but what constitutes "sufficient" only becomes clear after an initial analysis.
STACIE's documentation contains a section on [preparing inputs](../preparing_inputs/index.md)
to help you with this process.

Finally, we encourage you to delve into the [theory](../theory/index.md) behind STACIE.
Although we try to make STACIE usable without a full understanding of the technical details,
a good understanding will help you get the most out of it.
