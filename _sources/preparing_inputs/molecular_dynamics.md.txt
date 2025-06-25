# Recommendations for MD Simulations

## Finite Size Effects

Transport properties derived from {term}`MD` simulations of periodic systems
can be affected by finite-size effects.
Finite-size effects are particularly significant for diffusion coefficients.
This systematic error is known to be proportional to $1/L$,
where $L$ is the length scale of the simulation box.
The $1/L$ dependence allows for extrapolation to infinite box size by linear regression
or by applying analytical corrections, such as the Yeh-Hummer correction
{cite:p}`yeh_2004_system,maginn_2020_best`.

## Choice of Ensemble

The {term}`NVE` ensemble is generally recommended for computing transport coefficients,
as thermostats and barostats (used for simulations in the {term}`NVT` and {term}`NpT` ensembles)
can interfere with system dynamics and introduce bias in transport properties
{cite:p}`maginn_2020_best`.
For production runs, the NpT ensemble has an additional drawback:
barostats introduce coordinate scaling,
which directly perturbs the atomic mean squared displacements.

A good approach is to first equilibrate the system using NVT or NpT,
before switching to NVE for transport property calculations.
The main difficulty is that a single NVE simulation does not fully represent an NVT or NpT ensemble,
even if the average temperature and pressure match perfectly.
(They become equivalent in the thermodynamic limit, but we always simulate finite systems.)

NVE simulations lack the proper variance in the kinetic energy and/or volume.
This issue can be addressed by performing an ensemble of independent NVE simulations that are,
as a whole, representative of the NVT or NpT ensemble.
Practically, this can be achieved by first performing multiple NVT or NpT equilibration runs,
depending on the ensemble of interest.
The final state of each equilibration run then serves as a starting point for an NVE run,
**without rescaling the volume or kinetic energy**,
since rescaling to the mean would artificially lower the variance in these quantities.

Note that correctly simulating multiple independent NVE runs can be technically challenging.
It is not a widely used approach, not all MD codes are properly tested for it,
and the default settings of some MD codes are not suitable for NVE simulations.
Hence, one must always carefully check the validity of the simulations:

- First, check the conserved quantity (total energy) for drift or large fluctuations.
  Compared to the fluctuations of the kinetic energy, these deviations should be small.
- For the NVE simulations as a whole, the temperature distribution should be
  consistent with the NVT or NpT ensemble.
- Even if the NVE runs are performed correctly,
  one must ensure that the number of NVE runs is large enough
  to obtain a representative sample of the total energy distribution.

An additional challenge is the complexity of the MD workflow
with restarts in different ensembles and multiple independent runs.
All examples in the STACIE documentation work with NVE production runs,
show how to manage the workflow and validate the temperature distribution in detail.

## Thermostat and Barostat Settings

For the equilibration runs discussed above,
the choice of thermostat and barostat time constants is not critical,
as long as the algorithms are valid (i.e., no Berendsen thermo- or barostats)
and the simulations are long enough to allow for full equilibration of the system
within the equilibration run.
A local thermostat can be used to make the equilibration more efficient.

In some cases, e.g., to remain consistent with historical results,
or because some of the challenges of NVE simulations cannot be overcome,
one may still prefer to run production runs for transport properties in the NVT ensemble.
When you start a new project, however, always consider using NVE production runs.
If you must use NVT, studies suggest that well-tuned NVT simulations
yield comparable results to NVE simulations
{cite:p}`fanourgakis_2012_determining, basconi_2013_effects, ke_2022_effects`.
Basconi *et al.* recommended using a thermostat with slow relaxation times, global coupling,
and continuous rescaling (as opposed to random force contributions) {cite:p}`basconi_2013_effects`.
These are typically the opposite of the settings that are used for efficient equilibration runs.
A drawback of slow relaxation times is that longer simulations are required
to fully sample the correct ensemble.

## Block Averages

As discussed in the [block averages](block_averages.md) section,
the use of block averages is recommended for storing simulation data.
In the case of MD simulations, a safe initial block size is 10 time steps.
Usually, the integration time step in MD is small enough to ensure that the fastest oscillations
are sampled with 10 steps per period.
It is unlikely that transport properties are affected by the dynamics at shorter time scales,
so a block size of 10 time steps is a good starting point.
Once you have performed an initial analysis of the data,
you can adjust (increase) the block size further to optimize the data storage.
If you take multiples of 10, it is easy to reprocess the initial block averages
and convert them to averages over larger blocks.
