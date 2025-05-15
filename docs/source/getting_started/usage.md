# How to Use?

This section provides a bird's eye view on how to use Stacie in general.
More detailed information can be found in the remaining sections of the documentation.

The algorithms in Stacie are robust enough
to provide reliable estimates of autocorrelation integrals without user invention.
There is no need for extensive tweaking of its settings through a graphical user interface.
Instead, you pass the input data as a NumPy array and a few additional parameters
(such as a time step) to the functions implemented in the `stacie` library.
You can do this in a Jupyter Notebook for interactive work or in a Python script.
The input to Stacie are time series data on an equidistant time axis grid.
You can provide multiple independent sequences of exactly the same length to reduce uncertainties.
The analysis returns a `result` object including the following attributes:

- `acint`: The integral of the autocorrelation function.
- `corrtime_exp`: The *exponential* autocorrelation time.
- `corrtime_int`: The *integrated* autocorrelation time.

The standard errors of these estimates are accessible through the
`acint_std`, `corrtime_exp_std` and `corrtime_int_std` attributes,
respectively.
In addition, intermediate results of the analysis can be accessed,
e.g. to create plots using the built-in plotting functions.
We also provide a convenient message-pack serialization for storing results,
or you can store them using your favorite Python library.

Many properties are defined in terms of an autocorrelation integral.
They require slightly different settings and preprocessing of the input data.
This documentation contains instructions for
[the properties we have tested](../theory/properties/index.md).
In addition, we provide [worked examples](../examples/index.md)
that show in detail how Stacie is used in practice.

If you plan to produce publication-grade research with Stacie,
the analysis inevitably becomes a two-step process.
The main difficulty is determining the length of the input time series,
which requires some prior knowledge.
As explained in the section on [autocorrelation time](../theory/properties/autocorrelation_time.md),
your results will only be reliable if the length of the input sequences is much longer
than the *exponential* autocorrelation time.
This is a chicken-and-egg problem,
because you need Stacie's analysis to determine the autocorrelation time!
To solve this problem, you first run a preliminary analysis with initial input sequences,
make a rough estimate of the required sequence length,
and then generate the final input data and run the final analysis.
(If your initial input is much too short, you may even have to repeat this a few times.)
In extreme cases, you may need sequences with billions of steps,
making it impractical to store them in full detail prior to analysis.
To reduce the storage requirements, you can store
[block averages](../theory/preparing_inputs/block_averages.md).
The recommended block size is also related to the *exponential* autocorrelation time.
Therefore, some preliminary analysis may be required to prepare appropriate inputs.

Finally, we encourage you to delve into the [theory](../theory/index.md) behind Stacie.
Although we try to make Stacie usable without a full understanding of the technical details,
a good understanding will help you to get the most out of it.
