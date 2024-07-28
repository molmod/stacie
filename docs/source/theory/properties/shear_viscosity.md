# Shear Viscosity

The shear viscosity of a fluid is related to the autocorrelation
of microscopic off-diagonal pressure tensor fluctuations as follows:

$$
    \eta = \frac{V}{2 k_\text{B} T}
           \int_{-\infty}^{+\infty}
           \cov[P_{xy}(t_0), P_{xy}(t_0 + \Delta_t)]\,\mathrm{d}\Delta_t
$$

where $V$ is the volume of the simulation cell,
$k_\text{B}$ is the Boltzmann constant,
$T$ is the temperature,
and $P_{xy}$ is an off-diagonal pressure tensor element.
The time origin $t_0$ is arbitrary:
the expectation value is computed over all possible time origins.

The derivation of this result can be found in several references, e.g.,
Appendix C.3 of "Understanding Molecular Simulation"
by Frenkel and Smit {cite:p}`frenkel_2002_understanding`,
Section 8.4 of "Theory of Simple Liquids"
by Hansen and McDonald {cite:p}`hansen_2013_theory`,
or Section 13.3 of "Statistical Mechanics: Theory and Molecular Simulation"
by Tuckerman {cite:p}`tuckerman_2023_statistical`.


## Five Independent Off-diagonal Pressure Components of an Isotropic Liquid

There are few references explaining how to compute
five independent components of the pressure tensor {cite:p}`daivis_1994_comparison`
in the case of an isotropic fluid.
We will derive this result below, as it provides a useful recipe for computations.

To facilitate working with linear transformations of pressure tensors,
we use Voigt notation:

$$
    \mathbf{P} =
    \Bigl[
        \begin{matrix}
            P_{xx} & P_{yy} & P_{zz} & P_{yz} & P_{zx} & P_{xy}
        \end{matrix}
    \Bigr]^\top
$$

The transformation to the traceless form then becomes
$\mathbf{\tilde{P}} = \mathbf{T} \mathbf{P}$ with:

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

\begin{align*}
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
\end{align*}

The zero eigenvalue corresponds to the isotropic component being removed.
Transforming the pressure to this basis of eigenvectors constructs five off-diagonal components.
Since this transformation is orthonormal, the five components remain statistically uncorrelated.
It can be shown that the two first off-diagonal components must be rescaled
with a factor $1/\sqrt{2}$, as in $\mathbf{P}^\prime = \mathbf{V} \mathbf{P}$ with

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

to obtain five time-dependent off-diagonal pressure components
that can be used as inputs to the viscosity calculation:

$$
    \eta =
    \frac{V}{2 k_\text{B} T}
    \int_{-\infty}^{+\infty}
    \cov[
        P_i^{\prime}(t_0) P_i^\prime(t_0 + \Delta_t)
    ]
    \,\mathrm{d}\Delta_t
    \qquad
    \forall\,i\in\{1,2,3,4,5\}
$$

For the last three components, this result is trivial.
The second component, $P'_2$, is found by rotating the Cartesian axes $45^\circ$ about the $x$-axis.

$$
    \mathbf{R} &= \left[\begin{matrix}
        1 & & \\
        & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\
        & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}
    \end{matrix}\right]
    \\
    \mathbf{P} &= \left[\begin{matrix}
        P_{xx} & P_{xy} & P_{zx} \\
        P_{xy} & P_{yy} & P_{yz} \\
        P_{zx} & P_{yz} & P_{zz}
    \end{matrix}\right]
    \\
    \mathbf{R}\mathbf{P}\mathbf{R}^\top &= \left[\begin{matrix}
        p_{xx} &
        \frac{\sqrt{2} P_{xy}}{2} - \frac{\sqrt{2} P_{zx}}{2} &
        \frac{\sqrt{2} P_{xy}}{2} + \frac{\sqrt{2} P_{zx}}{2}
        \\
        \frac{\sqrt{2} P_{xy}}{2} - \frac{\sqrt{2} P_{zx}}{2} &
        \frac{P_{yy}}{2} - P_{yz} + \frac{P_{zz}}{2} &
        \frac{P_{yy}}{2} - \frac{P_{zz}}{2}
        \\
        \frac{\sqrt{2} P_{xy}}{2} + \frac{\sqrt{2} P_{zx}}{2} &
        \frac{P_{yy}}{2} - \frac{P_{zz}}{2} &
        \frac{P_{yy}}{2} + P_{yz} + \frac{P_{zz}}{2}
    \end{matrix}\right]
$$

In the new axes frame, the last off-diagonal element is a proper off-diagonal term equal to
$\frac{P_{yy}}{2}-\frac{P_{zz}}{2}$.

For the first component, $P'_1$, the proof is a little more involved.
There is no rotation of the Cartesian axis frame
in which this linear combination appears as an off-diagonal element.
Instead, it is simply a scaled sum of two off-diagonal stresses:

$$
    P'_1 = \alpha\left(
        P_{xx} - \frac{P_{yy}}{2} - \frac{P_{zz}}{2}
    \right) = \alpha\left(
        \frac{P_{xx}}{2} - \frac{P_{yy}}{2}
    \right) + \alpha\left(
        \frac{P_{xx}}{2} - \frac{P_{zz}}{2}
    \right)
$$

By working out the autocorrelation functions of $P'_1$ and $P'_2$ one finds that,
for the case of an isotropic liquid,
they have the same expectation values if $\alpha=\frac{1}{\sqrt{3}}$.
First expand the covariances:

\begin{align*}
&\begin{aligned}
    \cov[P'_1(t_0), P'_1(\Delta_t)] =
    &\,
        - \frac{\alpha^2}{2} \cov[P_{xx}(t_0), P_{yy}(t_0+\Delta_t)]
        - \frac{\alpha^2}{2} \cov[P_{yy}(t_0), P_{xx}(t_0+\Delta_t)]
    \\
    &\,
        - \frac{\alpha^2}{2} \cov[P_{xx}(t_0), P_{zz}(t_0+\Delta_t)]
        - \frac{\alpha^2}{2} \cov[P_{zz}(t_0), P_{xx}(t_0+\Delta_t)]
    \\
    &\,
        + \frac{\alpha^2}{4} \cov[P_{yy}(t_0), P_{zz}(t_0+\Delta_t)]
        + \frac{\alpha^2}{4} \cov[P_{zz}(t_0), P_{xx}(t_0+\Delta_t)]
    \\
    &\,
        + \frac{\alpha^2}{4} \cov[P_{yy}(t_0), P_{yy}(t_0+\Delta_t)]
        + \frac{\alpha^2}{4} \cov[P_{zz}(t_0), P_{zz}(t_0+\Delta_t)]
    \\
    &\,
        + \alpha^2 \cov[P_{xx}(t_0), P_{xx}(t_0+\Delta_t)]
\end{aligned}
\\
&\begin{aligned}
    \cov[P'_2(t_0), P'_2(\Delta_t)] =
    &\,
        - \frac{1}{4} \cov[P_{yy}(t_0), P_{zz}(t_0+\Delta_t)]
        - \frac{1}{4} \cov[P_{zz}(t_0), P_{yy}(t_0+\Delta_t)]
    \\
    &\,
        + \frac{1}{4} \cov[P_{yy}(t_0), P_{yy}(t_0+\Delta_t)]
        + \frac{1}{4} \cov[P_{zz}(t_0), P_{zz}(t_0+\Delta_t)]
\end{aligned}
\end{align*}

Because the liquid is isotropic, permutations of Cartesian axes do not affect the expectations values, which greatly simplifies both expressions:

$$
    \cov[P'_1(t_0), P'_1(\Delta_t)] &=
        \frac{3\alpha^2}{2} \cov[P_{xx}(t_0), P_{xx}(t_0+\Delta_t)]
        - \frac{3\alpha^2}{2} \cov[P_{xx}(t_0), P_{yy}(t_0+\Delta_t)]
    \\
    \cov[P'_2(t_0), P'_2(\Delta_t)] &=
        \frac{1}{2} \cov[P_{xx}(t_0), P_{xx}(t_0+\Delta_t)]
        - \frac{1}{2} \cov[P_{xx}(t_0), P_{yy}(t_0+\Delta_t)]
$$

These two expectation values are consistent when $\alpha^2 = 1/3$.

Using the same expansion technique, without permutations, one can show (after a lot of writing)
that the average viscosity over the five components proposed here is equivalent to
the equation proposed by Daivis and Evans {cite:p}`daivis_1994_comparison`:

$$
    \eta = \frac{1}{10} \frac{V}{2k_\text{B} T} \int_{-\infty}^{+\infty}
        \mean\left[\mathbf{\tilde{P}}(t_0):\mathbf{\tilde{P}}(t_0 + \Delta_t)\right]
        \,\mathrm{d}\Delta_t
$$

(This is Eq. A5 in their paper rewritten in the current notation.)
Working with the five components, as we propose, is advantageous because it makes explicit
how many independent sequences are used as input, which allows precise uncertainty quantification.

## How to Compute with Stacie

It is assumed that you can load the time-dependent pressure tensor components
(diagonal and off-diagonal) into a NumPy array `pcomps`.
Each row of this array corresponds to one pressure tensor component
in the order $P_{xx}$, $P_{yy}$, $P_{zz}$, $P_{zx}$, $P_{yz}$, $P_{xy}$.
(Same order as in Voigt notation.)
The columns correspond to time steps.
You also need to store the cell volume, temperature,
Boltzmann constant, and time step in Python variables,
all in consistent units.
With these requirements, the viscosity can be computed as follows:

```python
import numpy as np
from stacie import compute_spectrum, estimate_acint, plot_results

# Load all the required inputs, the details of which will depend on your use case.
pcomps = ...
volume, temperature, boltzmann_const, timestep = ...

# Convert pressure components to five independent components.
# This is the optimal usage of pressure information
# and it informs Stacie of the amount of independent inputs.
indep_pcomps = np.array([
    (pcomps[0] - 0.5 * pcoms[1] - 0.5 * pcoms[2]) / np.sqrt(3),
    0.5 * pcoms[1] - 0.5 * pcoms[2],
    pcomps[3],
    pcomps[4],
    pcomps[5],
])

# Actual computation with Stacie.
spectrum = compute_spectrum(
    indep_pcomps,
    prefactor=0.5 * volume / (temperature * boltzmann_const),
    timestep=timestep,
)
result = estimate_acint(spectrum)
print("Shear viscosity", result.props["acint"])
print("Uncertainty of the shear viscosity", result.props["acint_std"])

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

This script is trivially extended to combine data from multiple trajectories.
