# Diffusion Coefficient

The diffusion coefficient (or diffusivity) of a set of $N$ particles in $D$ dimensions is given by:

$$
    D = \frac{1}{2ND}\int_{-\infty}^{+\infty}
        \sum_{n=1}^N \sum_{d=1}^D
        \cov[v_{n,d}(t_0), v_{n,d}(t_0 + \Delta_t)]\,\mathrm{d}\Delta_t
$$

where $v_{n,d}(t)$ is the Cartesian component $d$ of the time-dependent velocity of particle $n$.
If the particles are molecules, their center of mass velocities can be used.

For a simple fluid, the result is called the self-diffusion coefficient or self-diffusivity.
The same expression applies to the diffusion coefficient of components of a mixture
or guest molecules in porous media.

This definition is valid only if the particles of interest exhibit diffusive motion.
If they oscillate around a fixed center,
the zero-frequency component of the velocity autocorrelation spectrum will converge to zero,
implying that the diffusion coefficient is zero.
This can be the apparent result when the diffusion is governed by an activated hopping process
and the simulation is too short to include such rare events.

The derivation of this result can be found in several references, e.g.,
Section 4.4.1 of "Understanding Molecular Simulation"
by Frenkel and Smit {cite:p}`frenkel_2002_understanding`,
Section 7.7 of "Theory of Simple Liquids"
by Hansen and McDonald {cite:p}`hansen_2013_theory`,
or Section 13.3 of "Statistical Mechanics: Theory and Molecular Simulation"
by Tuckerman {cite:p}`tuckerman_2023_statistical`.


## How to Compute with Stacie

It is assumed that you can load the particle velocities into a NumPy array `velocities`.
Each row of this array corresponds to the Cartesian velocity component of a particle.
The columns correspond to time steps.
You also need to store the time step in a Python variable.
The diffusion coefficient can then be computed as follows:

```python
import numpy as np
from stacie import compute_spectrum, estimate_acint, plot_results

# Load all the required inputs, the details of which will depend on your use case.
velocities = ...
timestep = ...

# Computation with Stacie.
# Note that the factor 1/(N*D) is implied:
# the average spectrum over all velocity components is computed.
spectrum = compute_spectrum(
    velocities,
    prefactor=0.5,
    timestep=timestep,
)
result = estimate_acint(spectrum)
print("Diffusion coefficient", result.props["acint"])
print("Uncertainty of the diffusion coefficient", result.props["acint_std"])

# The unit configuration assumes SI units are used systematically.
# You may need to adapt this to the units of your data.
uc = UnitConfig(
    acint_unit_str="m$^2$/s",
    time_unit=1e-12,
    time_unit_str="ps",
    freq_unit=1e12,
    freq_unit_str="THz",
)
plot_results("diffusion_coefficient.pdf", result, uc)
```

For more details, check out the example notebook: [Diffusion on a Surface with Newtonian Dynamics](../../examples/surface_diffusion.py).
