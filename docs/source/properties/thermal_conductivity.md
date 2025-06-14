# Thermal Conductivity

The thermal conductivity of a system is related to the autocorrelation
of the heat flux as follows:

$$
    \kappa = \frac{1}{V k_\text{B} T^2}
        \frac{1}{3}\sum_{\alpha=x, y, z}
        \frac{1}{2}
        \int_{-\infty}^{+\infty}
        \cov[\hat{J}^\text{h}_\alpha(t_0) \,,\, \hat{J}^\text{h}_\alpha(t_0 + \Delta_t)]
        \,\mathrm{d}\Delta_t
$$

where $V$ is the volume of the simulation cell,
$k_\text{B}$ is the Boltzmann constant,
$T$ is the temperature,
and $\hat{J}^\text{h}_\alpha$ is the instantaneous heat flux along one of the Cartesian directions.
The time origin $t_0$ is arbitrary:
the expected value is computed over all possible time origins.

The derivation of this result can be found in
Section 8.5 of "Theory of Simple Liquids"
by Hansen and McDonald {cite:p}`hansen_2013_theory`.

:::{warning}
The LAMMPS `compute/heat flux` command is reported to produce unphysical results
when many-body interactions (e.g., angle, dihedral, impropers) are present
{cite:p}`jamali_2019_octp`, {cite:p}`surblys_2019_application`,
{cite:p}`boone_2019_heat`, {cite:p}`surblys_2021_methodology`.
This command only treats pairwise interactions correctly.
If this is relevant, one should use the `compute heat/flux` command with
[`compute centroid/stress/atom`](https://docs.lammps.org/compute_heat_flux.html).
For systems with only two-body interactions,
the `compute heat/flux` command with the `compute stress/atom` command is sufficient.
Molecular liquids are practically always simulated with some many-body terms,
and thus require the `compute centroid/stress/atom` command.
:::

## How to Compute with STACIE?

It is assumed that you can load the time-dependent heat flux components
into a 2D NumPy array `heatflux`.
Each row of this array corresponds to one heat flux component
in the order $\hat{J}_x$, $\hat{J}_y$, and $\hat{J}_z$.
Columns correspond to time steps.
You also need to store the cell volume, temperature,
Boltzmann constant, and time step in Python variables,
all in consistent units.
With these requirements, the thermal conductivity can be computed as follows:

```python
import numpy as np
from stacie import compute_spectrum, estimate_acint, plot_results, PadeModel, UnitConfig

# Load all the required inputs, the details of which will depend on your use case.
heatflux = ...
volume, temperature, boltzmann_const, timestep = ...

# Actual computation with STACIE.
# Note that the average spectrum over the three components is implicit.
# There is no need to include 1/3 here.
spectrum = compute_spectrum(
    heatflux,
    prefactors=1.0 / (volume * temperature**2 * boltzmann_const),
    timestep=timestep,
)
result = estimate_acint(spectrum, PadeModel([0, 2], [2]))
print("Thermal conductivity", result.acint)
print("Uncertainty of the thermal conductivity", result.acint_std)

# The unit configuration assumes SI units are used systematically.
# You may need to adapt this to the units of your data.
uc = UnitConfig(
    acint_symbol="κ",
    acint_unit_str="W m$^{-1}$ K$^{-1}$",
    time_unit=1e-12,
    time_unit_str="ps",
    freq_unit=1e12,
    freq_unit_str="THz",
)
plot_results("thermal_conductivity.pdf", result, uc)
```

This script is trivially extended to combine data from multiple trajectories.

A worked example can be found in the notebook
[Thermal Conductivity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)](../examples/lj_thermal_conductivity.py).
