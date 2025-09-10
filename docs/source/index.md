# Welcome to STACIE's Documentation

STACIE is a Python package and algorithm that computes time integrals of autocorrelation functions.
It is primarily designed for post-processing molecular dynamics simulations.
However, it can also be used for more general analysis of time-correlated data.
Typical applications include estimating transport properties
and the uncertainty of averages over time-correlated data,
as well as analyzing characteristic timescales.

```{only} html
![Graphical Summary](static/github_repo_card_dark.png)
```

```{only} latex
![Graphical Summary](static/github_repo_card_light.png)
```

STACIE is developed in the context of a collaboration between
the [Center for Molecular Modeling](https://molmod.ugent.be/)
and the tribology group of [Labo Soete](https://www.ugent.be/ea/emsme/en/research/soete)
at [Ghent University](https://ugent.be/).
STACIE is open-source software (LGPL-v3 license) and is available on
[GitHub](https://github.com/molmod/stacie) and [PyPI](https://pypi.org/project/stacie).

```{only} html
This online documentation provides practical instructions on how to use STACIE,
as well as the theoretical background needed to understand what STACIE computes and how it works.
We also provide [a PDF version of the documentation](https://molmod.github.io/stacie/documentation.pdf).
```

```{only} latex
This is a PDF version of the online documentation of STACIE.
The latest version of the documentation can be found at <https://molmod.github.io/stacie/>.
```

Please cite the following in any publication that relies on STACIE:

> Gözdenur, T.; Fauconnier, D.; Verstraelen, T. "STable AutoCorrelation Integral Estimator (STACIE):
> Robust and accurate transport properties from molecular dynamics simulations"
> arXiv 2025, [arXiv:2506.20438](https://arxiv.org/abs/2506.20438).

This manuscript has been submitted to The Journal of Chemical Information and Modeling
and the citation records will be updated when appropriate.

A follow-up paper is nearly completed that will describe in detail the calculation of shear viscosity
with STACIE:

> Gözdenur, T.; Fauconnier, D.; Verstraelen, T. "Reliable Viscosity Calculation from High-Pressure
> Equilibrium Molecular Dynamics: Case Study of 2,2,4-Trimethylhexane.", in preparation.

In addition, we are preparing another follow-up paper showing how to estimate
diffusion coefficients with proper uncertainty quantification using STACIE,
which is currently not fully documented yet.

Copy-pasteable citation records in various formats are provided in [](getting_started/cite.md).

```{toctree}
:hidden:

getting_started/index.md
theory/index.md
preparing_inputs/index.md
properties/index.md
examples/index.md
references.md
glossary.md
development/index.md
code_of_conduct.md
```
