# Properties

This section outlines the statistical or physical quantities
that can be computed as the integral of an autocorrelation function.
For each property, a code skeleton is provided
as a starting point for your calculations.

First, we discuss few properties that may be relevant to several scientific disciplines:

- [The standard error of the mean of time-correlated data](error_estimates.md)
- The exponential and integrated [autocorrelation time](autocorrelation_time.md)

The following physicochemical properties can be computed
as autocorrelation integrals of outputs of molecular dynamics simulations,
using so-called Green-Kubo relations
{cite:p}`green_1952_markoff,green_1954_markoff,kubo_1957_statistical,helfand_1960_transport`.
These properties are sometimes referred to as diagonal transport coefficients {cite:p}`pegolo_2025_transport`.
- [Diffusion coefficient](diffusion_coefficient.md), $D$
- [Electrical conductivity](electrical_conductivity.md), $\sigma$
- [Thermal conductivity](thermal_conductivity.md), $\kappa$
- [Shear viscosity](shear_viscosity.md), $\eta$
- [Bulk viscosity](bulk_viscosity.md), $\eta_\text{bulk}$

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

## Some general recommendations for MD simulations to compute transport properties

### 1. Finite size effects

Transport properties in molecular dynamics (MD) simulations of periodic systems can be
affected by finite size effects. To assess these, one can run simulations with varying box sizes
and analyze properties e.g. as a function of inverse box size (1/$L$).
In particular, finite size effects are particularly are known to be significant for diffusion coefficients.
Extrapolating to the infinite box limit or applying analytical corrections,
such as the Yeh-Hummer correction {cite:p}`yeh_2004_system,maginn_2020_best`,
can help mitigate these effects.

### 2. Choice of ensemble

The NVE ensemble is generally recommended for computing transport coefficients, as thermostats
(used in NVT and NpT ensembles) can interfere with system dynamics and introduce bias.
A good approach would be to first equilibrate the system using NpT (to determine equilibrium density)
followed by NVT, before switching to NVE for transport property calculations. One should always verify
that the average pressure and temperature remain close to the desired values during the NVE run(s).

### 3. Thermostats and barostats

Although NVE is generally recommended, transport properties are often computed in NVT ensembles.
However, thermostats and barostats can introduce systematic errors, particularly if their relaxation times
are too short. Studies {cite:p}`fanourgakis_2012_determining, basconi_2013_effects, ke_2022_effects` suggest that NVE and
well-tuned NVT simulations yield comparable results. Basconi et al. recommend a thermostat with slow
relaxation times, global coupling, and continuous rescaling (as opposed to random force contributions)
{cite:p}`basconi_2013_effects`. For example, a Nosé-Hoover thermostat with a relaxation
time of at least 1 ps should be good for minimal interference. One can rule out such errors more
rigorously by running multiple simulations with systematically increased thermostat (and barostat)
relaxation times.
