# Ionic Electrical Conductivity

The ionic electrical conductivity of a system is related to the autocorrelation
of the charge current as follows:

$$
    \sigma = \frac{1}{V k_\text{B} T}
        \frac{1}{d}\sum_{i=1}^d
        \frac{1}{2}
        \int_{-\infty}^{+\infty}
        \cov[\hat{J}^\text{c}_i(t_0) \,,\, \hat{J}^\text{c}_i(t_0 + \Delta_t)]
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
it can also be derived from the velocities ($\hat{\mathbf{v}}_n(t)$)
and the net charges ($q_n$) of the charge carriers as follows:

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
result = estimate_acint(spectrum, ExpPolyModel([0, 1, 2]))
print("Electrical conductivity", result.acint)
print("Uncertainty of the electrical conductivity", result.acint_std)

# The unit configuration assumes SI units are used systematically.
# You may need to adapt this to the units of your data.
uc = UnitConfig(
    acint_unit_str="S m$^{-1}$",
    time_unit=1e-12,
    time_unit_str="ps",
    freq_unit=1e12,
    freq_unit_str="THz",
)
plot_results("electrical_conductivity.pdf", result, uc)
```

There are several ways to alter this script, depending on your needs and the available data:

- This script is trivially extended to combine data from multiple trajectories.
- Some codes can directly output the charge current,
  which will reduce the amount of data stored on disk.
- Some simulation codes will print out the instantaneous dipole moment,
  to which finite differences can be applied to compute the charge current.
  Even if the dipole moment is printed only every $B$ steps,
  this approximation is useful and corresponds to taking block averages of the charge current.
  See the section on [block averages](../preparing_inputs/block_averages.md)
  for more details.

A worked example can be found in the notebook
[Ionic Conductivity and Self-diffusivity in Molten Sodium Chloride at 1100 K (OpenMM)](../examples/molten_salt.py)
