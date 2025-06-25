# Ionic Electrical Conductivity

The ionic electrical conductivity of a system is related to the autocorrelation
of the charge current as follows:

$$
    \sigma = \frac{1}{V k_\text{B} T}
        \frac{1}{d}\sum_{\alpha=1}^d
        \frac{1}{2}
        \int_{-\infty}^{+\infty}
        \cov[\hat{J}^\text{c}_\alpha(t_0) \,,\, \hat{J}^\text{c}_\alpha(t_0 + \Delta_t)]
        \,\mathrm{d}\Delta_t
$$

where $V$ is the volume of the simulation cell,
$k_\text{B}$ is the Boltzmann constant,
$T$ is the temperature,
$d$ is the dimensionality of the system,
and $\hat{J}^\text{c}_i$ is the instantaneous charge current along one of the Cartesian directions.
The time origin $t_0$ is arbitrary:
the expected value is computed over all possible time origins.

The derivation of this result can be found in
Appendix C.3.1 of "Understanding Molecular Simulation"
by Frenkel and Smit {cite:p}`frenkel_2002_understanding`,
or Section 7.7 of "Theory of Simple Liquids"
by Hansen and McDonald {cite:p}`hansen_2013_theory`.

If your simulation code does not print out the charge current,
it can also be derived from the velocities, $\hat{\mathbf{v}}_n(t)$,
and the net charges, $q_n$, of the charge carriers as follows:

$$
    \hat{\mathbf{J}}(t) = \sum_{n=1}^{N_q} q_n \hat{\mathbf{v}}_n(t)
$$

where $N_q$ is the number of charge carriers.
The charge current can also be interpreted as
the time derivative of the instantaneous dipole moment of the system.

In the case of molecular ions, the center-of-mass velocity can be used, but this is not critical.
You will get the same conductivity (possibly with slightly larger uncertainties)
when using the velocity of any single atom in a molecular ion instead.
The charges of ions must be integer multiples of the elementary charge
{cite:p}`grasselli_2019_topological`.

## Nernst-Einstein Approximation

The electrical conductivity is related to the (correlated) diffusion of the charge carriers.
When correlations between the ions are neglected, one obtains the Nernst-Einstein approximation
of the conductivity in terms of the self-diffusion coefficients of the ions.
We include the derivation here because a consistent treatment of the pre-factors
can be challenging.
(Literature references are not always consistent due to differences in notation.)
Our derivation is general, i.e., for an arbitrary number of different *types*
of charge carriers, which are not restricted to monovalent ions.

First, insert the expression for the charge current into the conductivity
and then bring the sums out of the integral:

$$
    \sigma = \frac{1}{V k_\text{B} T}
        \frac{1}{d}\sum_{i=1}^d
        \sum_{n=1}^{N_q} \sum_{m=1}^{N_q}
        q_n q_m
        \frac{1}{2}
        \int_{-\infty}^{+\infty}
        \cov[\hat{v}_{n,i}(t_0) \,,\, \hat{v}_{m,i}(t_0 + \Delta_t)]
        \,\mathrm{d}\Delta_t
$$

In the Nernst-Einstein approximation,
all correlations between ion velocities (even of the same type) are neglected
by discarding all off-diagonal terms in the double sum over $n$ and $m$.

$$
    \sigma \approx \sigma_{NE} = \frac{1}{V k_\text{B} T}
        \sum_{n=1}^{N_q}
        q_n^2
        \frac{1}{d}\sum_{i=1}^d
        \frac{1}{2}
        \int_{-\infty}^{+\infty}
        \cov[\hat{v}_{n,i}(t_0) \,,\, \hat{v}_{n,i}(t_0 + \Delta_t)]
        \,\mathrm{d}\Delta_t
$$

To further connect this equation to diffusion coefficients,
the number of *types* of charge carriers is called $K$.
Each type $k \in \{1, \ldots, K\}$ has a set of ions $S_k$ with charge $q_k$.
The number of ions in each set is $N_k=|S_k|$.
With these conventions, we can rewrite the equation as:

$$
    \sigma_{NE} = \frac{1}{V k_\text{B} T}
        \sum_{k=1}^{K}
        q_k^2 N_k
        \left(
        \frac{1}{N_k d}\sum_{i=1}^d
        \sum_{n\in S_k}
        \frac{1}{2}
        \int_{-\infty}^{+\infty}
        \cov[\hat{v}_{n,i}(t_0) \,,\, \hat{v}_{n,i}(t_0 + \Delta_t)]
        \,\mathrm{d}\Delta_t
        \right)
$$

The part between parentheses is the self-diffusion coefficient of the ions of type $k$.
Finally, we get:

$$
    \sigma_\text{NE} = \frac{1}{k_\text{B}T} \sum_{k=1}^{K} q_k^2 \rho_k D_k
$$

where $\rho_k$ and $D_k$ are the concentration and the diffusion coefficient of charge carrier $k$,
respectively.
The Nernst-Einstein approximation may not seem useful
because it neglects correlated motion between different types of charge carriers.
(The effect may be large!)
Nevertheless, a comparison of the Nernst-Einstein approximation to the actual conductivity
can help to quantify the degree of such correlations.
{cite:p}`shao_2020_role`

## How to Compute with STACIE?

It is assumed that you can load the time-dependent ion velocity components
into a NumPy array `ionvels`.
In the example below, this is a three-index array,
where the first index is for the ion, the second for the Cartesian component,
and the last for the time step.
To compute the charge current, you need to put the charges of the ions
in an array `charges`.
You also need to store the cell volume, temperature,
Boltzmann constant, and time step in Python variables,
all in consistent units.
With these requirements, the ionic electrical conductivity can be computed as follows:

```python
import numpy as np
from stacie import compute_spectrum, estimate_acint, plot_results, ExpPolyModel, UnitConfig

# Load all the required inputs, the details of which will depend on your use case.
# We assume ionvels has shape `(nstep, natom, ncart)`
# and charges is a 1D array with shape `(natom,)`
ionvels = ...
charges = ...
volume, temperature, boltzmann_const, timestep = ...

# Compute the charge current
chargecurrent = np.einsum("ijk,j->ki", ionvels, charges)

# Actual computation with STACIE.
# Note that the average spectrum over the three components is implicit.
# There is no need to include 1/3 here.
# Note that the zero-frequency component is usually not reliable
# because usually the total momentum is constrained or conserved.
spectrum = compute_spectrum(
    chargecurrent,
    prefactors=1.0 / (volume * temperature * boltzmann_const),
    timestep=timestep,
    include_zero_freq=False,
)
# The unit configuration assumes SI units are used systematically.
# You may need to adapt this to the units of your data.
uc = UnitConfig(
    acint_unit_str="S m$^{-1}$",
    time_unit=1e-12,
    time_unit_str="ps",
    freq_unit=1e12,
    freq_unit_str="THz",
)
# Actual analysis with STACIE.
result = estimate_acint(spectrum, ExpPolyModel([0, 1, 2]), unit_config=uc, verbose=True)
print("Electrical conductivity", result.acint)
print("Uncertainty of the electrical conductivity", result.acint_std)

plot_results("electrical_conductivity.pdf", result, uc)
```

A common scenario is that you have the positions of the ions instead of their velocities,
or the dipole moment of the system.
Even if this information is stored with large time intervals,
you can still compute the ionic conductivity by first computing a
[block-averaged](../preparing_inputs/block_averages.md) charge current.

$$
    \bar{\mathbf{J}}^\text{c}_{i+1/2}
      = \frac{1}{\Delta_t}
        \int_{t_i}^{t_{i+1}} \hat{\mathbf{J}}^\text{c}(t)\,\mathrm{d}t
      = \frac{\hat{\mathbf{p}}_{i+1} - \hat{\mathbf{p}}_i}{\Delta_t}
$$

where $\hat{\mathbf{p}}_i$ is the dipole moment at time step $i$.
Formally, this resembles a finite difference approximation of the charge current,
except that $\Delta_t$ can be large.

:::{warning}
If you have sampled velocities or charge currents directly with a large sampling interval,
they cannot be used because they are not proper [block averages](../preparing_inputs/block_averages.md).
In this case, your data violates the [Niquist-Shannon sampling theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem),
and the results will be perturbed by aliasing artifacts.
The block averages satisfy the sampling theorem by construction
because they average out the high-frequency components.
:::

For example, if the dipole moment is stored every 10 ps in eÅ,
you can compute the charge current as follows:

```python
import numpy as np
from scipy.constants import value
from stacie import compute_spectrum, estimate_acint, plot_results, ExpPolyModel, UnitConfig

# Values of units of external data in "internal" SI base units.
ELCHARGE = value("elementary charge")
PICOSECOND = 1e-12
ANGSTROM = 1e-10
TERAHERTZ = 1e12
BOLTZMANN_CONST = value("Boltzmann constant")

# Load all the required inputs, the details of which will depend on your use case.
# It is assumed that each row of `dipoles` corresponds to a single Cartesian component
# and each column to a time step.
dipoles = (...) * ELCHARGE * ANGSTROM
timestep = 10.0 * PICOSECOND
volume, temperature  = ...

# Compute charge current, as if you are using finite difference approximation.
chargecurrent = np.diff(dipoles, axis=1) / timestep

# Computation with STACIE, as before.
spectrum = compute_spectrum(
    chargecurrent,
    prefactors=1.0 / (volume * temperature * BOLTZMANN_CONST),
    timestep=timestep,
    include_zero_freq=False,
)
uc = UnitConfig(
    acint_unit_str="S m$^{-1}$",
    time_unit=PICOSECOND,
    time_unit_str="ps",
    freq_unit=TERAHERTZ,
    freq_unit_str="THz",
)
result = estimate_acint(spectrum, ExpPolyModel([0, 1, 2]), verbose=True, unit_config=uc)
plot_results("electrical_conductivity.pdf", result, uc)
```

A worked example can be found in the notebook
[Ionic Conductivity and Self-diffusivity in Molten Sodium Chloride at 1100 K (OpenMM)](../examples/molten_salt.py)
