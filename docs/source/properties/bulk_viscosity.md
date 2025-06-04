# Bulk Viscosity

The bulk viscosity of a fluid is related to the autocorrelation
of isotropic pressure fluctuations as follows:

$$
    \eta_b = \frac{V}{k_\text{B} T} \frac{1}{2}
        \int_{-\infty}^{+\infty}
        \cov[\hat{P}_\text{iso}(t_0) \,,\, \hat{P}_\text{iso}(t_0 + \Delta_t)]\,\mathrm{d}\Delta_t
$$

where $V$ represents the volume of the simulation cell,
$k_\text{B}$ is the Boltzmann constant,
$T$ is the temperature,
and $\hat{P}_\text{iso}$ is the instantaneous isotropic pressure.
The time origin $t_0$ is arbitrary:
the expected value is computed over all possible time origins.

The derivation of this result can be found in several references, e.g.,
Section 8.5 of "Theory of Simple Liquids"
by Hansen and McDonald {cite:p}`hansen_2013_theory`,
or Section 2.7 of "Computer Simulation of Liquids"
by Allen and Tildesley {cite:p}`allen_2017_computer`.

As will be shown below, one must take into account that the average pressure is not zero.
For STACIE, there is no need to subtract the average pressure first.
Instead, you can simply drop the DC component from the spectrum.

## How to Compute with STACIE?

It is assumed that you can load the diagonal, time-dependent pressure tensor components
into a 2D NumPy array `pcomps`.
(The same array as for [shear viscosity](shear_viscosity.md) can be used.)
Each row of this array corresponds to one pressure tensor component in the order
$\hat{P}_{xx}$, $\hat{P}_{yy}$, $\hat{P}_{zz}$, $\hat{P}_{zx}$, $\hat{P}_{yz}$, $\hat{P}_{xy}$.
(This is the same order as in Voigt notation. The last three components are not used and can be omitted.)
Columns correspond to time steps.
You also need to store the cell volume, temperature,
Boltzmann constant, and time step in Python variables,
all in consistent units.
With these requirements, the bulk viscosity can be computed as follows:

```python
import numpy as np
from stacie import compute_spectrum, estimate_acint, plot_results, PadeModel, UnitConfig

# Load all the required inputs, the details of which will depend on your use case.
pcomps = ...
volume, temperature, boltzmann_const, timestep = ...

# Convert pressure components to the isotropic pressure
piso = (pcomps[0] + pcomps[1] + pcomps[2]) / 3

# Actual computation with STACIE.
spectrum = compute_spectrum(
    piso,
    prefactors=volume / (temperature * boltzmann_const),
    timestep=timestep,
    include_zero_freq=False,
)
result = estimate_acint(spectrum, PadeModel([0, 2], [2]))
print("Bulk viscosity:", result.acint)
print("Uncertainty of the bulk viscosity:", result.acint_std)

# The unit configuration assumes SI units are used systematically.
# You may need to adapt this to the units of your data.
uc = UnitConfig(
    acint_unit_str="Pa s",
    time_unit=1e-12,
    time_unit_str="ps",
    freq_unit=1e12,
    freq_unit_str="THz",
)
plot_results("bulk_viscosity.pdf", result, uc)
```

This script can be trivially extended to combine data from multiple trajectories.

A worked example can be found in the notebook
[Bulk viscosity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)](../examples/lj_bulk_viscosity.py)
