# Recommendations for MD simulations

## 1. Finite size effects

Transport properties derived from {term}`MD` simulations of periodic systems
can be affected by finite size effects.
Finite size effects are particularly significant for diffusion coefficients.
This systematic error is known to be proportional to $1/L$,
where $L$ is the length scale of the simulation box.
The $1/L$ dependence allows for extrapolation to infinite box size by linear regression
or by applying analytical corrections, such as the Yeh-Hummer correction
{cite:p}`yeh_2004_system,maginn_2020_best`.

## 2. Choice of ensemble

The NVE ensemble is generally recommended for computing transport coefficients,
as thermostats and barostats (used in NVT and NpT ensembles)
can interfere with system dynamics and introduce bias in transport properties
{cite:p}`maginn_2020_best`.
For production runs, the NpT ensemble has an additional drawback:
barostat implementations introduce coordinate scaling,
which directly perturbs the atomic mean squared displacements.

A good approach is to first equilibrate the system using NVT or NpT,
before switching to NVE for transport property calculations.
The main difficulty is that a single NVE simulation does not fully represent an NVT or NpT ensemble,
even if the average temperature and pressure match perfectly.
Such a simulation lacks the proper variance in the kinetic energy and/or volume.
This issue can be addressed by performing an ensemble of independent NVE simulations that are,
as a whole, representative of the NVT or NpT ensemble.
Practically, this can be achieved by performing multiple NVT or NpT equilibration runs,
depending on the ensemble of interest.
The final state of each equilibration run then serves as a starting point for an NVE run,
**without rescaling the volume or kinetic energy**,
since that would artificially lower the variance in these quantities.
All examples in the STACIE documentation follow this approach.

## 3. Thermostat and barostat settings

For the equilibration runs discussed in the previous section,
the choice of thermostat and barostat time constants is not critical,
as long as the algorithms are valid (so no Berendsen thermo- or barostats)
and the simulations are long enough to allow for full equilibration of the system
within the equilibration run.
A local thermostat can be used to make the equilibration more efficient.

In some cases, e.g. to remain consistent with historical results,
one may still prefer to run production runs for transport properties in the NVT ensemble.
When you start a new project, however, there are no good excuses for not using NVE.
If you really must use NVT,
there are studies suggesting that well-tuned NVT simulations yield comparable results to NVE simulations.
{cite:p}`fanourgakis_2012_determining, basconi_2013_effects, ke_2022_effects`
Basconi *et al.* recommended using a thermostat with slow relaxation times, global coupling,
and continuous rescaling (as opposed to random force contributions) {cite:p}`basconi_2013_effects`.
A drawback of slow relaxation times is that longer simulations are required
to fully sample the correct ensemble.
