# How to Use?

This section provides a bird's eye view on how to use STACIE in general.
More detailed information can be found in the remaining sections of the documentation.

The algorithms in STACIE are robust enough
to provide reliable estimates of autocorrelation integrals without user invention.
There is no need for extensive tweaking of its settings through a graphical user interface.
Instead, you provide the time-correlated input data as a NumPy array,
together with a few additional parameters (such as a time step),
to the functions implemented in the `stacie` library.
You can do this in a Jupyter Notebook for interactive work or in a Python script.

The most important inputs for STACIE are time series data on an equidistant time axis grid.
You can provide multiple independent sequences of exactly the same length to reduce uncertainties.
The analysis returns a `result` object including the following attributes:

- `acint`: The integral of the autocorrelation function.
- `corrtime_int`: The *integrated* autocorrelation time.
- `corrtime_exp`: The *exponential* autocorrelation time (if supported by the selected model).

The estimated standard errors of these estimates are accessible through the
`acint_std`, `corrtime_int_std` and `corrtime_exp_std` attributes,
respectively.
In addition, intermediate results of the analysis can be accessed,
e.g. to create plots using the built-in plotting functions.

Many properties are defined in terms of an autocorrelation integral.
They require slightly different settings and preprocessing of the input data.
STACIE's documentation contains instructions for
[the properties we have tested](../theory/properties/index.md).
In addition, we provide [worked examples](../examples/index.md)
that show in detail how STACIE is used in practice.

If you plan to produce publication-grade research with STACIE,
the analysis inevitably becomes an iterative process.
The main difficulty is to provide sufficient data for the analysis,
but it only becomes clear after the first analysis what "sufficient" means.
STACIE's documentation contains a section on [preparing inputs](../theory/preparing_inputs/index.md)
to help you with this process.

Finally, we encourage you to delve into the [theory](../theory/index.md) behind STACIE.
Although we try to make STACIE usable without a full understanding of the technical details,
a good understanding will help you to get the most out of it.
