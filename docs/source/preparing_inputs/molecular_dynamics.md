# Recommendations for MD Simulations

## Finite Size Effects

Transport properties derived from {term}`MD` simulations of periodic systems
can be affected by finite-size effects.
They are particularly pronounced for diffusion coefficients.
This systematic error is known to be proportional to $1/L$,
where $L$ is the length scale of the simulation box {cite:p}`yeh_2004_system`.
The $1/L$ dependence allows for extrapolation to infinite box size by linear regression
{cite:p}`jamali_2018_shear`
or by applying analytical corrections, such as the Yeh-Hummer correction
{cite:p}`yeh_2004_system,maginn_2020_best`.

## Choice of Ensemble

The {term}`NVE` ensemble is in principle recommended for computing transport coefficients,
because thermostats and barostats
(used for simulations in the {term}`NVT` and {term}`NpT` ensembles)
can interfere with system dynamics and introduce bias in transport properties
{cite:p}`maginn_2020_best`.
For production runs, the NpT ensemble has an additional drawback:
barostats introduce coordinate scaling,
which directly perturbs the atomic mean squared displacements.

Despite this, NVE production runs face several challenges in practice.
They must be preceded by NVT or NpT equilibration runs
to ensure that the system is at the desired temperature and pressure,
which is also needed for NVT production runs.
A first difficulty is that a single NVE simulation does not fully represent an NVT or NpT ensemble,
even if the average temperature and pressure match perfectly.
(They become equivalent in the thermodynamic limit,
but the system sizes in molecular dynamics are far from this limit.)
NVE simulations lack the proper variance in the kinetic energy and/or volume.
This issue can be addressed by performing an ensemble of independent NVE simulations that are,
as a whole, representative of the NVT or NpT ensemble.
Furthermore, NVE simulations often show a small drift in the conserved quantity,
depending on the time step, cutoffs, constraint tolerances and numerical precision.
A systematic heating of the system then biases the transport properties.

Setting up a valid ensemble of independent NVE simulations is an extra hurdle.
Performing all production runs at exactly the same energy and/or volume,
e.g. the average of a set of equilibration runs, should be avoided.
A better approach is to let each production run start from the final state
of its own (independent) equilibration run, without rescaling the volume or the kinetic energy.
Without rescaling, this is effectively a Monte Carlo sampling of states in the NVT or NpT ensemble,
with one NVE simulation per state.
Alternatively, the random sampling can be replaced
by a deterministic grid of total energies and/or volumes,
constructed with STACIE's [Designed Quadrature](../examples/dq.py) module,
such that a simple average over the NVE runs reproduces the NVT or NpT ensemble average
with a smaller variance than Monte Carlo sampling with the same number of runs.
In both approaches, the set of NVE runs as a whole represents the NVT or NpT ensemble,
while each individual run retains unperturbed Newtonian dynamics.

Due to the challenges outlined above, NVE simulations are not widely used for transport properties.
If you decide to follow this route, always carefully validate the simulations:

- First, check the conserved quantity (total energy) for drift or large fluctuations.
  Both should be small compared to the fluctuations of the kinetic energy.
  When divided by the number of atoms,
  all errors of the conserved quantity should be a small fraction of $k_\text{B} T$.
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

When NVE production runs are not feasible or desirable,
one can also perform NVT production runs.
With well-tuned thermostat settings,
NVT simulations can yield results that are consistent with NVE
{cite:p}`fanourgakis_2012_determining, basconi_2013_effects, ke_2022_effects`.
Basconi *et al.* recommended using a thermostat with a slow relaxation time, global coupling,
and velocity rescaling {cite:p}`basconi_2013_effects`.
These are typically the opposite of the settings that are used for efficient equilibration runs,
e.g., fast relaxation times, local coupling, and stochastic forces.
A potential drawback of slow relaxation times is that long simulations may be required
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
