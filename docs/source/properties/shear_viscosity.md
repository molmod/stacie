# Shear Viscosity

The shear viscosity of a fluid is related to the autocorrelation
of microscopic off-diagonal pressure tensor fluctuations as follows:

$$
    \eta = \frac{V}{k_\text{B} T} \frac{1}{2}
        \int_{-\infty}^{+\infty}
        \cov[\hat{P}_{xy}(t_0) \,,\, \hat{P}_{xy}(t_0 + \Delta_t)]\,\mathrm{d}\Delta_t
$$

where $V$ is the volume of the simulation cell,
$k_\text{B}$ is the Boltzmann constant,
$T$ is the temperature,
and $\hat{P}_{xy}$ is an instantaneous off-diagonal pressure tensor element.
The time origin $t_0$ is arbitrary:
the expected value is computed over all possible time origins.

The derivation of this result can be found in several references, e.g.,
Appendix C.3.2 of "Understanding Molecular Simulation"
by Frenkel and Smit {cite:p}`frenkel_2002_understanding`,
Section 8.4 of "Theory of Simple Liquids"
by Hansen and McDonald {cite:p}`hansen_2013_theory`,
or Section 13.3.1 of "Statistical Mechanics: Theory and Molecular Simulation"
by Tuckerman {cite:p}`tuckerman_2023_statistical`.

## Five Independent Anisotropic Pressure Contributions of an Isotropic Liquid

To the best of our knowledge, there is no prior work demonstrating how to prepare
five *independent* inputs with anisotropic pressure tensor contributions
that can be used as inputs to the autocorrelation integral.
For instance, the result below is not mentioned in a recent comparison of methods
for incorporating diagonal elements of the traceless pressure tensor
{cite:p}`mercier_2023_computation`.
Since a pressure tensor has six degrees of freedom,
one of which corresponds to the isotropic pressure,
the remaining five should be associated with anisotropic contributions.

It is well known that the viscosity of an isotropic fluid can be derived from
six off-diagonal and diagonal traceless pressure tensor elements
{cite:p}`daivis_1994_comparison`.
However, by subtracting the isotropic term, the six components of the traceless pressure tensor
become statistically correlated.
For a proper {term}`uncertainty` analysis of the estimated viscosity,
STACIE requires the inputs to be statistically independent,
so the Daivis and Evans equation cannot be directly used.
Here, we provide a transformation of the pressure tensor
that yields five independent contributions,
each of which can be used individually to compute the viscosity.
The average of these five viscosities is equivalent to the result of Daivis and Evans.

To facilitate working with linear transformations of pressure tensors, we adopt Voigt notation:

$$
    \hat{\mathbf{P}} =
    \Bigl[
        \begin{matrix}
            \hat{P}_{xx} & \hat{P}_{yy} & \hat{P}_{zz} & \hat{P}_{yz} & \hat{P}_{zx} & \hat{P}_{xy}
        \end{matrix}
    \Bigr]^\top
$$

The transformation to the traceless form then becomes
$\hat{\mathbf{P}}_\text{tl} = \mathbf{T} \hat{\mathbf{P}}$ with:

$$
    \mathbf{T} =
    \left[\begin{matrix}
        \frac{2}{3} & -\frac{1}{3} & -\frac{1}{3} & & &
        \\
        -\frac{1}{3} & \frac{2}{3} & -\frac{1}{3} & & &
        \\
        -\frac{1}{3} & -\frac{1}{3} & \frac{2}{3} & & &
        \\
        & & & 1 & &
        \\
        & & & & 1 &
        \\
        & & & & & 1
   \end{matrix}\right]
$$

This symmetric matrix is an idempotent projection matrix and has an eigendecomposition
$\mathbf{T}=\mathbf{U}\mathbf{\Lambda}\mathbf{U}^\top$ with:

$$
    \begin{aligned}
    \operatorname{diag}(\mathbf{\Lambda}) &=
    \left[\begin{matrix}
        0 \\ 1 \\ 1 \\ 1 \\ 1 \\ 1
    \end{matrix}\right]
    &
    \mathbf{U} &=
    \left[\begin{matrix}
        \frac{1}{\sqrt{3}} & \sqrt{\frac{2}{3}} & 0 & & &
        \\
        \frac{1}{\sqrt{3}} & -\frac{1}{\sqrt{6}} & \frac{1}{\sqrt{2}} & & &
        \\
        \frac{1}{\sqrt{3}} & -\frac{1}{\sqrt{6}} & -\frac{1}{\sqrt{2}} & & &
        \\
        & & & 1 & &
        \\
        & & & & 1 &
        \\
        & & & & & 1
    \end{matrix}\right]
    \end{aligned}
$$

The zero eigenvalue corresponds to the isotropic component being removed.
Transforming the pressure tensor to this eigenvector basis constructs five anisotropic components.
Since this transformation is orthonormal, the five components remain statistically uncorrelated.
It can be shown that the first two anisotropic components must be rescaled by a factor of $1/\sqrt{2}$,
as in $\hat{\mathbf{P}}^\prime = \mathbf{V} \hat{\mathbf{P}}$, with:

$$
    \mathbf{V} =
    \left[\begin{matrix}
        \frac{1}{\sqrt{3}} & 0 & & &
        \\
        -\frac{1}{2\sqrt{3}} & \frac{1}{2} & & &
        \\
        -\frac{1}{2\sqrt{3}} & -\frac{1}{2} & & &
        \\
        & & 1 & &
        \\
        & & & 1 &
        \\
        & & & & 1
    \end{matrix}\right]
$$

to obtain five time-dependent anisotropic pressure components
that can be used as inputs to the viscosity calculation:

$$
    \eta =
    \frac{V}{k_\text{B} T} \frac{1}{2}
    \int_{-\infty}^{+\infty}
    \cov[\hat{P}_i^{\prime}(t_0) \,,\, \hat{P}_i^\prime(t_0 + \Delta_t)]
    \,\mathrm{d}\Delta_t
    \qquad
    \forall\,i\in\{1,2,3,4,5\}
$$

For the last three components, this result is trivial.
The second component, $\hat{P}'_2$, is found by rotating the Cartesian axes $45^\circ$ about the $x$-axis.

$$
    \mathcal{R} &= \left[\begin{matrix}
        1 & & \\
        & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\
        & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}
    \end{matrix}\right]
    \\
    \hat{\mathcal{P}} &= \left[\begin{matrix}
        \hat{P}_{xx} & \hat{P}_{xy} & \hat{P}_{zx} \\
        \hat{P}_{xy} & \hat{P}_{yy} & \hat{P}_{yz} \\
        \hat{P}_{zx} & \hat{P}_{yz} & \hat{P}_{zz}
    \end{matrix}\right]
    \\
    \mathbf{R}\hat{\mathcal{P}}\mathbf{R}^\top &= \left[\begin{matrix}
        \hat{P}_{xx} &
        \frac{\sqrt{2} \hat{P}_{xy}}{2} - \frac{\sqrt{2} \hat{P}_{zx}}{2} &
        \frac{\sqrt{2} \hat{P}_{xy}}{2} + \frac{\sqrt{2} \hat{P}_{zx}}{2}
        \\
        \frac{\sqrt{2} \hat{P}_{xy}}{2} - \frac{\sqrt{2} \hat{P}_{zx}}{2} &
        \frac{\hat{P}_{yy}}{2} - \hat{P}_{yz} + \frac{\hat{P}_{zz}}{2} &
        \frac{\hat{P}_{yy}}{2} - \frac{\hat{P}_{zz}}{2}
        \\
        \frac{\sqrt{2} \hat{P}_{xy}}{2} + \frac{\sqrt{2} \hat{P}_{zx}}{2} &
        \frac{\hat{P}_{yy}}{2} - \frac{\hat{P}_{zz}}{2} &
        \frac{\hat{P}_{yy}}{2} + \hat{P}_{yz} + \frac{\hat{P}_{zz}}{2}
    \end{matrix}\right]
$$

In the new axes frame, the last off-diagonal element is a proper anisotropic term, expressed as
$\frac{\hat{P}_{yy}}{2} - \frac{\hat{P}_{zz}}{2}$.

For the first component, $\hat{P}'_1$, the proof is slightly more intricate.
There is no rotation of the Cartesian axis frame
that results in this linear combination appearing as an off-diagonal element.
Instead, it is simply a scaled sum of two anisotropic stress components:

$$
    \hat{P}'_1 = \alpha\left(
        \hat{P}_{xx} - \frac{\hat{P}_{yy}}{2} - \frac{\hat{P}_{zz}}{2}
    \right) = \alpha\left(
        \frac{\hat{P}_{xx}}{2} - \frac{\hat{P}_{yy}}{2}
    \right) + \alpha\left(
        \frac{\hat{P}_{xx}}{2} - \frac{\hat{P}_{zz}}{2}
    \right)
$$

By working out the autocorrelation functions of $\hat{P}'_1$ and $\hat{P}'_2$ one finds that,
for the case of an isotropic liquid,
they have the same expected values if $\alpha=\frac{1}{\sqrt{3}}$.
First expand the covariances:

$$
    &\cov[\hat{P}'_1(t_0) \,,\, \hat{P}'_1(t_0 + \Delta_t)] =
    \\
    &\qquad
        - \frac{\alpha^2}{2} \cov[\hat{P}_{xx}(t_0) \,,\, \hat{P}_{yy}(t_0+\Delta_t)]
        - \frac{\alpha^2}{2} \cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{xx}(t_0+\Delta_t)]
    \\
    &\qquad
        - \frac{\alpha^2}{2} \cov[\hat{P}_{xx}(t_0) \,,\, \hat{P}_{zz}(t_0+\Delta_t)]
        - \frac{\alpha^2}{2} \cov[\hat{P}_{zz}(t_0) \,,\, \hat{P}_{xx}(t_0+\Delta_t)]
    \\
    &\qquad
        + \frac{\alpha^2}{4} \cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{zz}(t_0+\Delta_t)]
        + \frac{\alpha^2}{4} \cov[\hat{P}_{zz}(t_0) \,,\, \hat{P}_{xx}(t_0+\Delta_t)]
    \\
    &\qquad
        + \frac{\alpha^2}{4} \cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{yy}(t_0+\Delta_t)]
        + \frac{\alpha^2}{4} \cov[\hat{P}_{zz}(t_0) \,,\, \hat{P}_{zz}(t_0+\Delta_t)]
    \\
    &\qquad
        + \alpha^2 \cov[\hat{P}_{xx}(t_0) \,,\, \hat{P}_{xx}(t_0+\Delta_t)]
    \\[0.5em]
    &\cov[\hat{P}'_2(t_0) \,,\, \hat{P}'_2(t_0 + \Delta_t)] =
    \\
    &\qquad
        - \frac{1}{4} \cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{zz}(t_0+\Delta_t)]
        - \frac{1}{4} \cov[\hat{P}_{zz}(t_0) \,,\, \hat{P}_{yy}(t_0+\Delta_t)]
    \\
    &\qquad
        + \frac{1}{4} \cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{yy}(t_0+\Delta_t)]
        + \frac{1}{4} \cov[\hat{P}_{zz}(t_0) \,,\, \hat{P}_{zz}(t_0+\Delta_t)]
$$

Because the liquid is isotropic,
permutations of Cartesian axes do not affect the expected values,
which greatly simplifies the expressions.

$$
    &\cov[\hat{P}'_1(t_0), \hat{P}'_1(t_0 + \Delta_t)]
    \\
    &\qquad
        \frac{3\alpha^2}{2} \cov[\hat{P}_{xx}(t_0) \,,\, \hat{P}_{xx}(t_0+\Delta_t)]
    \\
    &\qquad
        - \frac{3\alpha^2}{4} \cov[\hat{P}_{xx}(t_0) \,,\, \hat{P}_{yy}(t_0+\Delta_t)]
        - \frac{3\alpha^2}{4} \cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{xx}(t_0+\Delta_t)]
    \\[0.5em]
    &\cov[\hat{P}'_2(t_0), \hat{P}'_2(t_0 + \Delta_t)] =
    \\
    &\qquad
        \frac{1}{2} \cov[\hat{P}_{xx}(t_0) \,,\, \hat{P}_{xx}(t_0+\Delta_t)]
    \\
    &\qquad
        - \frac{1}{4} \cov[\hat{P}_{xx}(t_0) \,,\, \hat{P}_{yy}(t_0+\Delta_t)]
        - \frac{1}{4} \cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{xx}(t_0+\Delta_t)]
$$

These two expected values are consistent when $\alpha^2 = 1/3$.

Using the same expansion technique,
it is shown below that the average viscosity over the five components proposed here
is equivalent to the equation proposed by Daivis and Evans {cite:p}`daivis_1994_comparison`:

$$
    \eta = \frac{1}{5} \frac{V}{k_\text{B} T} \frac{1}{2} \int_{-\infty}^{+\infty}
        \frac{1}{2}\mean\left[\hat{\mathbf{P}}_\text{tl}(t_0):\hat{\mathbf{P}}_\text{tl}(t_0 + \Delta_t)\right]
        \,\mathrm{d}\Delta_t
$$

(This is Eq. A5 in their paper rewritten in our notation.)
Working out the expansion, using
$\hat{P}_{\text{tl},xx} =
\frac{1}{3}(2\hat{P}_{xx} - \hat{P}_{yy} - \hat{P}_{zz})$
and similar definitions for the two other Cartesian components, we get:

$$
    &\frac{1}{2}\mean\left[
        \hat{\mathbf{P}}_\text{tl}(t_0):\hat{\mathbf{P}}_\text{tl}(t_0 + \Delta_t)
    \right] =
    \\
    &\qquad
        \cov[\hat{P}_{yz}(t_0) \,,\, \hat{P}_{yz}(t_0+\Delta_t)]
    \\
    &\qquad
        +\cov[\hat{P}_{zx}(t_0) \,,\, \hat{P}_{zx}(t_0+\Delta_t)]
    \\
    &\qquad
        +\cov[\hat{P}_{xy}(t_0) \,,\, \hat{P}_{xy}(t_0+\Delta_t)]
    \\
    &\qquad
        +\frac{1}{3}\cov[\hat{P}_{xx}(t_0) \,,\, \hat{P}_{xx}(t_0+\Delta_t)]
    \\
    &\qquad
        +\frac{1}{3}\cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{yy}(t_0+\Delta_t)]
    \\
    &\qquad
        +\frac{1}{3}\cov[\hat{P}_{zz}(t_0) \,,\, \hat{P}_{zz}(t_0+\Delta_t)]
    \\
    &\qquad
        -\frac{1}{6}\cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{zz}(t_0+\Delta_t)]
        -\frac{1}{6}\cov[\hat{P}_{zz}(t_0) \,,\, \hat{P}_{yy}(t_0+\Delta_t)]
    \\
    &\qquad
        -\frac{1}{6}\cov[\hat{P}_{zz}(t_0) \,,\, \hat{P}_{xx}(t_0+\Delta_t)]
        -\frac{1}{6}\cov[\hat{P}_{xx}(t_0) \,,\, \hat{P}_{zz}(t_0+\Delta_t)]
    \\
    &\qquad
        -\frac{1}{6}\cov[\hat{P}_{xx}(t_0) \,,\, \hat{P}_{yy}(t_0+\Delta_t)]
        -\frac{1}{6}\cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{xx}(t_0+\Delta_t)]
$$

We can do the same for our average viscosity over the five independent components:

$$
    \eta = \frac{1}{5} \frac{V}{k_\text{B} T} \frac{1}{2}
    \int_{-\infty}^{+\infty}
    \sum_{i=1}^5 \cov[\hat{P}_i^{\prime}(t_0) \,,\, \hat{P}_i^\prime(t_0 + \Delta_t)]
    \,\mathrm{d}\Delta_t
$$

Working out the expansion of the five terms in Cartesian pressure tensor components yields:

$$
    &\cov[ \hat{P}_1^{\prime}(t_0) \,,\, \hat{P}_1^\prime(t_0 + \Delta_t)] =
    \\
    &\qquad
        +\frac{1}{3}\cov[\hat{P}_{xx}(t_0) \,,\, \hat{P}_{xx}(t_0+\Delta_t)]
    \\
    &\qquad
        +\frac{1}{12}\cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{yy}(t_0+\Delta_t)]
    \\
    &\qquad
        +\frac{1}{12}\cov[\hat{P}_{zz}(t_0) \,,\, \hat{P}_{zz}(t_0+\Delta_t)]
    \\
    &\qquad
        +\frac{1}{12}\cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{zz}(t_0+\Delta_t)]
        +\frac{1}{12}\cov[\hat{P}_{zz}(t_0) \,,\, \hat{P}_{yy}(t_0+\Delta_t)]
    \\
    &\qquad
        -\frac{1}{6}\cov[\hat{P}_{zz}(t_0) \,,\, \hat{P}_{xx}(t_0+\Delta_t)]
        -\frac{1}{6}\cov[\hat{P}_{xx}(t_0) \,,\, \hat{P}_{zz}(t_0+\Delta_t)]
    \\
    &\qquad
        -\frac{1}{6}\cov[\hat{P}_{xx}(t_0) \,,\, \hat{P}_{yy}(t_0+\Delta_t)]
        -\frac{1}{6}\cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{xx}(t_0+\Delta_t)]
    \\[0.5em]
    &\cov[ \hat{P}_2^{\prime}(t_0) \,,\, \hat{P}_2^\prime(t_0 + \Delta_t)] =
    \\
    &\qquad
        +\frac{1}{4}\cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{yy}(t_0+\Delta_t)]
    \\
    &\qquad
        +\frac{1}{4}\cov[\hat{P}_{zz}(t_0) \,,\, \hat{P}_{zz}(t_0+\Delta_t)]
    \\
    &\qquad
        -\frac{1}{4}\cov[\hat{P}_{yy}(t_0) \,,\, \hat{P}_{zz}(t_0+\Delta_t)]
        -\frac{1}{4}\cov[\hat{P}_{zz}(t_0) \,,\, \hat{P}_{yy}(t_0+\Delta_t)]
    \\[0.5em]
    &\cov[ \hat{P}_3^{\prime}(t_0) \,,\, \hat{P}_3^\prime(t_0 + \Delta_t)] =
        \cov[\hat{P}_{yz}(t_0) \,,\, \hat{P}_{yz}(t_0+\Delta_t)]
    \\[0.5em]
    &\cov[ \hat{P}_4^{\prime}(t_0) \,,\, \hat{P}_4^\prime(t_0 + \Delta_t)] =
        \cov[\hat{P}_{zx}(t_0) \,,\, \hat{P}_{zx}(t_0+\Delta_t)]
    \\[0.5em]
    &\cov[ \hat{P}_5^{\prime}(t_0) \,,\, \hat{P}_5^\prime(t_0 + \Delta_t)] =
        \cov[\hat{P}_{xy}(t_0) \,,\, \hat{P}_{xy}(t_0+\Delta_t)]
$$

Adding these five contributions together
reproduces the exact same expansion as derived by Daivis and Evans.

Using the five anisotropic components, as proposed here, offers significant advantages.
It explicitly defines the number of independent sequences used as input,
enabling precise uncertainty quantification.

## How to Compute with STACIE?

It is assumed that you can load the time-dependent pressure tensor components
(diagonal and off-diagonal) into a 2D NumPy array `pcomps`.
Each row of this array corresponds to one pressure tensor component in the order
$\hat{P}_{xx}$, $\hat{P}_{yy}$, $\hat{P}_{zz}$, $\hat{P}_{zx}$, $\hat{P}_{yz}$, $\hat{P}_{xy}$
(same order as in Voigt notation).
Columns correspond to time steps.
You also need to store the cell volume, temperature, Boltzmann constant,
and time step in Python variables, all in consistent units.
With these requirements, the shear viscosity can be computed as follows:

```python
import numpy as np
from stacie import compute_spectrum, estimate_acint, plot_results, PadeModel, UnitConfig

# Load all the required inputs, the details of which will depend on your use case.
pcomps = ...
volume, temperature, boltzmann_const, timestep = ...

# Convert pressure components to five independent components.
# This is the optimal usage of pressure information
# and it informs STACIE of the number of independent inputs.
indep_pcomps = np.array([
    (pcomps[0] - 0.5 * pcomps[1] - 0.5 * pcomps[2]) / np.sqrt(3),
    0.5 * pcomps[1] - 0.5 * pcomps[2],
    pcomps[3],
    pcomps[4],
    pcomps[5],
])

# Actual computation with STACIE.
spectrum = compute_spectrum(
    indep_pcomps,
    prefactors=volume / (temperature * boltzmann_const),
    timestep=timestep,
)
result = estimate_acint(spectrum, PadeModel([0, 2], [2]))
print("Shear viscosity:", result.acint)
print("Uncertainty of the shear viscosity:", result.acint_std)

# The unit configuration assumes SI units are used systematically.
# You may need to adapt this to the units of your data.
uc = UnitConfig(
    acint_unit_str="Pa s",
    time_unit=1e-12,
    time_unit_str="ps",
    freq_unit=1e12,
    freq_unit_str="THz",
)
plot_results("shear_viscosity.pdf", result, uc)
```

This script can be trivially extended to combine data from multiple trajectories.

A worked example can be found in the notebook
[Shear viscosity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)](../examples/lj_shear_viscosity.py)
