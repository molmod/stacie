#!/usr/bin/env python3

# %% [markdown]
# # Diffusion on a Surface with Newtonian Dynamics
#
# This example shows how to compute the diffusion coefficient
# of a particle adsorbed on a crystal surface.
# For simplicity, the motion of the adsorbed particle is described
# by Newton's equations (without thermostat)
# and the particle can only move in two dimensions.
#
# This is a completely self-contained example that generates the input sequences
# (with numerical integration) and then analyzes them with STACIE.
# Atomic units are used unless otherwise noted.

# %% [markdown]
# ## Import Libraries and Configure `matplotlib`

# %%
import attrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import scipy.constants as sc
from numpy.typing import ArrayLike, NDArray
from stacie import UnitConfig, compute_spectrum, estimate_acint, ExpTailModel
from stacie.plot import plot_extras, plot_fitted_spectrum

# %%
mpl.rc_file("matplotlibrc")
# %config InlineBackend.figure_formats = ["svg"]

# %% [markdown]
# ## Data Generation

# %% [markdown]
# ### Potential energy surface
#
# The first cell below defines the potential energy of a particle on the surface
# and the force exerted by the surface on the particle.
# The model for the potential energy is a superposition of cosine functions:
#
# $$
#     U(\mathbf{r}) = -A \sum_{n=1}^N \cos(2\pi \mathbf{r}\cdot\mathbf{e}_n / \lambda)
# $$
#
# with
#
# $$
#     \mathbf{e}_n = \mathbf{e}_x \cos(n\alpha)  + \mathbf{e}_y \sin(n\alpha)
# $$
#
# The default settings for this notebook result in a hexagonal lattice:
# $A = 0.2\,\mathrm{eV}$,
# $\lambda = 5\,\mathrm{a}_0$,
# $N=3$, and
# $\alpha = 2\pi/3$.
# One may change these parameters to construct different types of surfaces:
#
# - A square lattice: $N = 2$ and $\alpha = \pi/2$.
# - A quasi-periodic pentagonal lattice: $N=5$ and $\alpha = 2\pi/5$.
#
# The remainder of the notebook is configured to work well for the default parameters.
# If you change the potential energy model, remaining settings also need to be adapted.

# %%
WAVELENGTH = 5.0
ALPHA = 2 * np.pi / 3
ANGLES = np.arange(3) * ALPHA
AMPLITUDE = 0.2 * sc.value("electron volt") / sc.value("atomic unit of energy")


def potential_energy_force(coords: ArrayLike) -> tuple[NDArray, NDArray]:
    """Compute the potential energies for given particle positions.

    Parameters
    ----------
    coords
        A NumPy array with one or more particle positions.
        The last dimension is assumed to have size two.
        Index 0 and 1 of the last axis correspond to x and y coordinates,
        respectively.

    Returns
    -------
    energy
        The potential energies for the given particle positions.
        An array with shape `pos.shape[:-1]`.
    force
        The forces acting on the particles.
        Same shape as `pos`, with same index conventions.
    """
    coords = np.asarray(coords, dtype=float)
    x = coords[..., 0]
    y = coords[..., 1]
    energy = 0
    force = np.zeros(coords.shape)
    wavenum = 2 * np.pi / WAVELENGTH
    for angle in ANGLES:
        arg = (x * np.cos(angle) + y * np.sin(angle)) * wavenum
        energy -= np.cos(arg)
        sin_wave = np.sin(arg) * wavenum
        force[..., 0] -= sin_wave * np.cos(angle)
        force[..., 1] -= sin_wave * np.sin(angle)
    return AMPLITUDE * energy, AMPLITUDE * force


# Quick visual test: the force is minus the energy gradient.
print(potential_energy_force([1, 2]))
print(nd.Gradient(lambda coords: potential_energy_force(coords)[0])([1, 2]))


# %%
def plot_pes():
    plt.close("pes")
    _, ax = plt.subplots(num="pes")
    xs = np.linspace(-30, 30, 201)
    ys = np.linspace(-30, 30, 201)
    coords = np.array(np.meshgrid(xs, ys)).transpose(1, 2, 0)
    energies = potential_energy_force(coords)[0]
    ax.contour(xs, ys, energies)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [a$_0$]")
    ax.set_ylabel("y [a$_0$]")
    ax.set_title("Potential Energy Surface")


plot_pes()

# %% [markdown]
# ### Newtonian Dynamics
#
# The following code cell implements a vectorized
# [Velocity Verlet integrator](https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet),
# which can integrate multiple independent trajectories at the same time.
# Some parameters, like mass and time step are fixed as global constants.
# The mass is that of an Argon atom converted to atomic units.
# The timestep is five femtosecond converted to atomic units.

# %%
MASS = sc.value("unified atomic mass unit") * 39.948 / sc.value("atomic unit of mass")
FEMTOSECOND = 1e-15 / sc.value("atomic unit of time")
PICOSECOND = 1e-12 / sc.value("atomic unit of time")
TERAHERTZ = 1e12 * sc.value("atomic unit of time")
TIMESTEP = 5 * FEMTOSECOND


@attrs.define
class Trajectory:
    """Bundle dynamics trajectory results.

    The first axis of all array attributes corresponds to time steps.
    """

    timestep: float = attrs.field()
    """The spacing between two recorded time steps."""

    coords: NDArray = attrs.field()
    """The time-dependent particle positions."""

    vels: NDArray = attrs.field()
    """The time-dependent particle velocities.

    If stride is larger than 1,
    this attribute contains the block-averaged velocity.
    """

    potential_energies: NDArray = attrs.field()
    """The time-dependent potential energies."""

    kinetic_energies: NDArray = attrs.field()
    """The time-dependent potential energies."""

    @classmethod
    def empty(cls, shape: tuple[int, ...], nstep: int, timestep: float):
        """Construct an empty trajectory object."""
        return cls(
            timestep,
            np.zeros((nstep, *shape, 2)),
            np.zeros((nstep, *shape, 2)),
            np.zeros((nstep, *shape)),
            np.zeros((nstep, *shape)),
        )

    @property
    def nstep(self) -> int:
        """The number of time steps."""
        return self.coords.shape[0]


def integrate(coords: ArrayLike, vels: ArrayLike, nstep: int, stride: int = 1):
    """Integrate Newton's equation of motion for the given initial conditions.

    Parameters
    ----------
    coords
        The initial particle positions.
        Index 0 and 1 of the last axis correspond to x and y coordinates.
    vels
        The initial particle velocities.
        Index 0 and 1 of the last axis correspond to x and y coordinates.
    nstep
        The number of MD time steps.
    stride
        The stride with which to record the trajectory data.

    Returns
    -------
    trajectory
        A Trajectory object holding all the results.
    """
    traj = Trajectory.empty(coords.shape[:-1], nstep // stride, TIMESTEP * stride)
    energies, forces = potential_energy_force(coords)
    delta_vels = forces * (0.5 * TIMESTEP / MASS)

    vels_block = 0
    for istep in range(traj.nstep * stride):
        vels += delta_vels
        coords += vels * TIMESTEP
        energies, forces = potential_energy_force(coords)
        delta_vels = forces * (0.5 * TIMESTEP / MASS)
        vels += delta_vels
        vels_block += vels
        if istep % stride == stride - 1:
            itraj = istep // stride
            traj.coords[itraj] = coords
            traj.vels[itraj] = vels_block / stride
            traj.potential_energies[itraj] = energies
            traj.kinetic_energies[itraj] = (0.5 * MASS) * (vels**2).sum(axis=-1)
            vels_block = 0
    return traj


def demo_energy_conservation():
    """Simple demo of the approximate energy conservation.

    The initial velocity is small enough
    to let the particle vibrate around the origin.
    """
    nstep = 100
    traj = integrate(np.zeros(2), np.full(2, 1e-4), nstep)
    plt.close("energy")
    _, ax = plt.subplots(num="energy")
    times = np.arange(traj.nstep) * traj.timestep
    ax.plot(times, traj.potential_energies, label="potential")
    ax.plot(times, traj.potential_energies + traj.kinetic_energies, label="total")
    ax.set_title("Energy Conservation Demo")
    ax.set_xlabel("Time [a.u. of time]")
    ax.set_ylabel(r"Energy [E$_\mathrm{h}$]")


demo_energy_conservation()

# %% [markdown]
# ### Demonstration of Deterministic Choas
#
# Newtonian dynamics is deterministic,
# but has chaotic solutions for many systems.
# The particle on a surface in this notebook is no exception.
# The following cell shows two trajectories
# for nearly identical initial conditions,
# but they slowly drift apart over time.
# After sufficient time,
# any information about their nearly identical initial conditions is lost.


# %%
def demo_chaos():
    vels = np.array([[1e-3, 1e-4], [1.00000001e-3, 1e-4]])
    traj = integrate(np.zeros((2, 2)), vels, 2500)
    plt.close("chaos")
    _, ax = plt.subplots(num="chaos")
    ax.plot(traj.coords[:, 0, 0], traj.coords[:, 0, 1], color="C1")
    ax.plot(traj.coords[:, 1, 0], traj.coords[:, 1, 1], color="C3", ls=":")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [a$_0$]")
    ax.set_ylabel("y [a$_0$]")
    ax.set_title("Two Trajectories")

    plt.close("chaos_dist")
    _, ax = plt.subplots(num="chaos_dist")
    times = np.arange(traj.nstep) * traj.timestep
    ax.semilogy(times, np.linalg.norm(traj.coords[:, 0] - traj.coords[:, 1], axis=-1))
    ax.set_xlabel("Time [a.u. of time]")
    ax.set_ylabel("Interparticle distance [a$_0$]")
    ax.set_title("Slow Separation")


demo_chaos()

# %% [markdown]
# Because the trajectories are
# [chaotic](https://en.wikipedia.org/wiki/Chaos_theory),
# the short term motion is ballistic,
# while the long term motion is a random walk.
# Note that the random walk is only found in a specific energy window.
# If the energy is too small,
# the particles will oscillate around a local potential energy minimum.
# If the energy is too large, or just high enough to cross barriers,
# the particles will follow almost linear paths over the surface.

# %% [markdown]
# ## Self-diffusion without block averages
#
# This section considers 100 independent particles whose initial velocities
# have the same magnitude by whose directions are random.
# Their velocities are used as inputs for STACIE
# to compute the diffusion coefficient.


# %%
def demo_stacie(stride=1):
    natom = 100
    nstep = 20000
    rng = np.random.default_rng(42)
    vels = rng.normal(0, 1, (natom, 2))
    vels *= 9.7e-4 / np.linalg.norm(vels, axis=1).reshape(-1, 1)
    traj = integrate(np.zeros((natom, 2)), vels, nstep, stride)

    plt.close(f"trajs_{stride}")
    _, ax = plt.subplots(num=f"trajs_{stride}")
    for i in range(natom):
        ax.plot(traj.coords[:, i, 0], traj.coords[:, i, 1])
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [a$_0$]")
    ax.set_ylabel("y [a$_0$]")
    ax.set_title(f"{natom} Newtonian Pseudo-Random Walks")

    spectrum = compute_spectrum(
        traj.vels.transpose(1, 2, 0).reshape(2 * natom, traj.nstep),
        timestep=traj.timestep,
    )
    uc = UnitConfig(
        acint_symbol="D",
        acint_unit=sc.value("atomic unit of time")
        / sc.value("atomic unit of length") ** 2,
        acint_unit_str="m$^2$ s",
        acint_fmt=".2e",
        freq_unit=TERAHERTZ,
        freq_unit_str="THz",
        time_unit=PICOSECOND,
        time_unit_str="ps",
        time_fmt=".2f",
    )

    # The maximum cutoff frequency is chosen to be 1 THz,
    # by inspecting the first spectrum plot.
    # Beyond the cutoff frequency, the spectrum has resonance peaks that
    # the ExpTailModel is not designed to handle.
    result = estimate_acint(
        spectrum, ExpTailModel(), fcut_max=TERAHERTZ, verbose=True, uc=uc
    )

    # Plotting
    plt.close(f"spectrum_{stride}")
    _, ax = plt.subplots(num=f"spectrum_{stride}")
    plot_fitted_spectrum(ax, uc, result)
    plt.close(f"extras_{stride}")
    _, axs = plt.subplots(2, 2, num=f"extras_{stride}")
    plot_extras(axs, uc, result)
    return result


# %%
result_1 = demo_stacie()

# %% [markdown]
# The spectrum has several peaks related to oscillations of the particles
# around a local minimum.
# These are irrelevant to the diffusion coefficient.
# The broad peak at zero frequency is used by STACIE
# to derive the diffusion coefficient.
# The value is not directly comparable to experiment
# because the 2D lattice model for the surface
# is not based on an experimental case.
# The order of magnitude is comparable to the self-diffusion constants
# of pure liquids {cite:p}`baba_2022_prediction`.
#
# It is also interesting to compare the integrated and exponential autocorrelation time,
# as they are not the same in this case.

# %%
print(f"corrtime_exp = {result_1.props['corrtime_exp'] / PICOSECOND:.3f} ps")
print(f"corrtime_int = {result_1.corrtime_int / PICOSECOND:.3f} ps")

# %% [markdown]
# The integrated autocorrelation time is smaller than
# the exponential autocorrelation time
# because the former represents an average of all time scales in the particle velocities.
# This includes the slow diffusion and faster oscillations in local minima.
# The exponential autocorrelation time only considers the slow diffusive motion.

# %% [markdown]
# ## Self-diffusion with block averages
#
# This section repeats the same example,
# but now with block averages of velocities.
# [Block averages](../theory/preparing_inputs/block_averages.md)
# are primarily useful for reducing storage requirements
# when saving trajectories to disk before processing them with STACIE.
# In this example, the block size is determined by the following guideline:

# %%
print(0.05 * result_1.props["corrtime_exp"] * np.pi / TIMESTEP)

# %% [markdown]
# Let's use a block size of 30 to stay on the safe side.

# %%
result_30 = demo_stacie(30)

# %% [markdown]
# As expected, there are not significant changes in the results.
#
# It is again interesting to compare the integrated and exponential autocorrelation times.

# %%
print(f"corrtime_exp = {result_30.props['corrtime_exp'] / PICOSECOND:.3f} ps")
print(f"corrtime_int = {result_30.corrtime_int / PICOSECOND:.3f} ps")

# %% [markdown]
# The exponential autocorrelation time is not affected by the block averages,
# but the integrated autocorrelation time has increased
# compared to the result without block averages.
# This is a consequence of the fact that
# the integrated correlation averages over all time scales in the input,
# and some of the fastest ones are washed out by the block averages.

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
acint_unit = sc.value("atomic unit of time") / sc.value("atomic unit of length") ** 2
acint_1 = result_1.acint / acint_unit
if abs(acint_1 - 5.85e-7) > 5e-9:
    raise ValueError(f"Wrong acint (no block average): {acint_1:.2e}")
acint_30 = result_30.acint / acint_unit
if abs(acint_30 - 5.87e-7) > 5e-9:
    raise ValueError(f"Wrong acint (block size 30): {acint_30:.2e}")
