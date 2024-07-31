# Properties

This section documents the statistical or physical quantities
that can be computed as the integral of an autocorrelation function.
For each property, a skeleton of sample code is provided
as a starting point for your calculations.

First, we discuss few properties that may be relevant to several scientific disciplines:

- [The standard error of the mean of time-correlated data](error_estimates.md)
- The exponential and integrated [autocorrelation time](autocorrelation_time.md)

The following physicochemical properties can be computed
as autocorrelation integrals of outputs of molecular dynamics simulations,
using so-called Green-Kubo relations
{cite:p}`green_1952_markoff,green_1954_markoff,kubo_1957_statistical,helfand_1960_transport`:

- [Diffusion coefficient](diffusion_coefficient.md)
- [Electrical conductivity](electrical_conductivity.md)
- [Thermal conductivity](thermal_conductivity.md)
- [Shear viscosity](shear_viscosity.md)
- [Bulk viscosity](bulk_viscosity.md)

```{toctree}
:hidden:

error_estimates.md
autocorrelation_time.md
diffusion_coefficient.md
electrical_conductivity.md
thermal_conductivity.md
shear_viscosity.md
bulk_viscosity.md
```

Notte that all transport properties estimated from molecular dynamics simulations
(of periodic systems) may be affected by finite size effects.
If relevant, one should extrapolate the results to an infinite box size
{cite:p}`yeh_2004_system`.
