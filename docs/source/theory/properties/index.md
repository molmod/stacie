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

## General recommendations for MD simulations to compute transport properties

### 1. Finite size effects

Transport properties in molecular dynamics (MD) simulations of periodic systems
can be affected by finite size effects.
Finite size effects are particularly significant for diffusion coefficients.
This systematic error is known to be proportional to $1/L$,
allowing for an extrapolation to infinite box size through linear regression
or by applying analytical corrections, such as the Yeh-Hummer correction
{cite:p}`yeh_2004_system,maginn_2020_best`.

### 2. Choice of ensemble

The NVE ensemble is generally recommended for computing transport coefficients,
as thermostats and barostats (used in NVT and NpT ensembles)
can interfere with system dynamics and introduce bias in the transport property.
A good approach would be to first equilibrate the system using NVT or NpT,
before switching to NVE for transport property calculations.
The main difficulty is that one NVE simulation is not representative for the NVT or NpT ensemble,
even if the average temperature and pressure coincides perfectly.
Such a simulation lacks the proper variance in the kinetic energy and/or volume.
This limitation can be solved by performing an ensemble of NVE simulations that are,
as a whole, representative for the NVT or NpT ensemble.
Practically, this can be accomplished by performing multiple NVT or NpT equilibration runs,
depending on the ensemble of interest.
The final state of the equilibration run is then used as starting point of the NVE run,
**without rescaling the volume or kinetic energy**,
since that would artifically lower the variance in these quantities.
All examples in the Stacie documentation use this technique.

### 3. Thermostat and barostat settings

For the equilibration runs discussed in the previous section,
the choice of the time constants of the thermo- and barostats is not critical,
as long as they allow for a full equilibration of this system
within the duration of the equilibration run.
During the equilibration, one can use a local thermostat to make the equilibration more effecient.

In some cases, one may still prefer to run production simulations for transport properties
in the NVT or NpT ensemble, despite the fact that this introduces an avoidable bias,
particularly if the relaxation times of the thermo- and/or barostat are too short.
Studies suggest that NVE and well-tuned NVT simulations yield comparable results.
{cite:p}`fanourgakis_2012_determining, basconi_2013_effects, ke_2022_effects`
Basconi et al. recommend a thermostat with slow relaxation times, global coupling,
and continuous rescaling (as opposed to random force contributions) {cite:p}`basconi_2013_effects`.
A disadvantage of slow relaxation times is that longer simulations are required
to fully sample the correct ensemble.
