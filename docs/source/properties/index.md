# Properties Derived from the Autocorrelation Function

This section outlines the statistical and physical quantities
that can be computed as the integral of an autocorrelation function.
For each property, a code skeleton is provided as a starting point for your calculations.
All skeletons assume that you can load the relevant input data into NumPy arrays.

First, we discuss a few properties that may be relevant to multiple scientific disciplines:

- [The uncertainty of the mean of time-correlated data](error_estimates.md)
- The exponential and integrated [autocorrelation time](autocorrelation_time.md)

The following physicochemical transport properties can be computed
as autocorrelation integrals of outputs from molecular dynamics simulations,
using the so-called Green-Kubo relations
{cite:p}`green_1952_markoff,green_1954_markoff,kubo_1957_statistical,helfand_1960_transport`.
These properties have recently been referred to as diagonal transport coefficients {cite:p}`pegolo_2025_transport`.

- [Shear viscosity](shear_viscosity.md), $\eta$
- [Bulk viscosity](bulk_viscosity.md), $\eta_b$
- [Thermal conductivity](thermal_conductivity.md), $\kappa$
- [Electrical conductivity](electrical_conductivity.md), $\sigma$
- [Diffusion coefficient](diffusion_coefficient.md), $D$

```{toctree}
:hidden:

error_estimates.md
autocorrelation_time.md
shear_viscosity.md
bulk_viscosity.md
thermal_conductivity.md
electrical_conductivity.md
diffusion_coefficient.md
```
