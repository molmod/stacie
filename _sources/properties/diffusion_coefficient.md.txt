# Diffusion Coefficient

The diffusion coefficient (or diffusivity) of a set of $N$ particles in $d$ dimensions is given by:

$$
D = \frac{1}{N\,d} \frac{1}{2}\int_{-\infty}^{+\infty}
    \sum_{n=1}^N \sum_{i=1}^{d}
    \cov[\hat{v}_{n,i}(t_0),\, \hat{v}_{n,i}(t_0 + \Delta_t)]\,\mathrm{d}\Delta_t
$$

where $\hat{v}_{n,i}(t)$ is the $i$-th Cartesian component of
the time-dependent velocity of particle $n$.
For molecular systems, the center-of-mass velocities are typically used.

For a simple fluid, the result is called the self-diffusion coefficient or self-diffusivity.
The same expression applies to the diffusion coefficient of components of a mixture
or guest molecules in porous media.

Note that this definition is valid only if the particles of interest exhibit diffusive motion.
If they oscillate around a fixed center,
the zero-frequency component of the velocity autocorrelation spectrum will approach zero,
resulting in a diffusion coefficient of zero.
This scenario may occur when the diffusion is governed by an activated hopping process,
and the simulation is too short to capture such rare events.

The derivation of this result can be found in several references, e.g.,
Section 4.4.1 of "Understanding Molecular Simulation"
by Frenkel and Smit {cite:p}`frenkel_2002_understanding`,
Section 7.7 of "Theory of Simple Liquids"
by Hansen and McDonald {cite:p}`hansen_2013_theory`,
or Section 13.3.2 of "Statistical Mechanics: Theory and Molecular Simulation"
by Tuckerman {cite:p}`tuckerman_2023_statistical`.

## How to Compute with STACIE?

It is assumed that you can load the particle velocities into a 2D NumPy array `velocities`.
Each row of this array corresponds to a single Cartesian component of a particle's velocity, while
each column corresponds to a specific time step.
You should also store the time step in a Python variable.
The diffusion coefficient can then be computed as follows:

```python
import numpy as np
from stacie import compute_spectrum, estimate_acint, plot_results, ExpPolyModel, UnitConfig

# Load all the required inputs, the details of which will depend on your use case.
velocities = ...
timestep = ...

# Computation with STACIE.
# Note that the factor 1/(N*d) is implied:
# the average spectrum over all velocity components is computed.
# Note that the zero-frequency component is usually not reliable
# because typically the total momentum is constrained or conserved.
spectrum = compute_spectrum(
    velocities,
    prefactors=1.0,
    timestep=timestep,
    include_zero_freq=False,
)
result = estimate_acint(spectrum, ExpPolyModel([0, 1, 2]))
print("Diffusion coefficient", result.acint)
print("Uncertainty of the diffusion coefficient", result.acint_std)

# The unit configuration assumes SI units are used systematically.
# You may need to adapt this to the units of your data.
uc = UnitConfig(
    acint_symbol="D",
    acint_unit_str="m$^2$/s",
    time_unit=1e-12,
    time_unit_str="ps",
    freq_unit=1e12,
    freq_unit_str="THz",
)
plot_results("diffusion_coefficient.pdf", result, uc)
```

One can also use particle positions and apply a finite difference approximation
to obtain the velocities.
(For trajectories obtained with a Verlet integrator, this does not introduce additional approximations.)
When positions are recorded every $B$ steps,
the finite difference approximation can also be applied.
The result is equivalent to block-averaging velocities and can thus be used as inputs for STACIE.
Consult the section on [block averages](../preparing_inputs/block_averages.md) for more details.

A worked example can be found in the notebook
[Diffusion on a Surface with Newtonian Dynamics](../examples/surface_diffusion.py).
