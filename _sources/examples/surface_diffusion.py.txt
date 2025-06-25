# %% [markdown]
# # Diffusion on a Surface with Newtonian Dynamics
#
# This example shows how to compute the diffusion coefficient
# of a particle adsorbed on a crystal surface.
# For simplicity, the motion of the adsorbed particle is described
# by Newton's equations (without thermostat), i.e. in the {term}`NVE` ensemble,
# and the particle can only move in two dimensions.
#
#
# This is a completely self-contained example that generates the input sequences
# (with numerical integration) and then analyzes them with STACIE.
# Unless otherwise noted, atomic units are used.

# %% [markdown]
# ## Library Imports and Matplotlib Configuration

# %%
import attrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import scipy.constants as sc
from numpy.typing import ArrayLike, NDArray
from stacie import (
    UnitConfig,
    compute_spectrum,
    estimate_acint,
    LorentzModel,
    plot_extras,
    plot_fitted_spectrum,
)

# %%
mpl.rc_file("matplotlibrc")
# %config InlineBackend.figure_formats = ["svg"]

# %% [markdown]
# ## Data Generation

# %% [markdown]
# ### Potential energy surface
#
# The first cell below defines the potential energy of a particle on a surface,
# as well as the force that the surface exerts on the particle.
# The potential energy model is a superposition of cosine functions:
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
# The rest of the notebook is set up to work well with the default parameters.
# If you change the potential energy model, remaining settings will also need to be adapted.

# %%
WAVELENGTH = 5.0
ALPHA = 2 * np.pi / 3
ANGLES = np.arange(3) * ALPHA
EV = sc.value("electron volt") / sc.value("atomic unit of energy")
AMPLITUDE = 0.2 * EV


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


# %% [markdown]
# The following code cell provides a quick visual test of the forces using
# [numdifftools](https://github.com/pbrod/numdifftools).
# (The force is equal to minus the energy gradient.)

# %%
print(potential_energy_force([1, 2]))
print(nd.Gradient(lambda coords: potential_energy_force(coords)[0])([1, 2]))

# %% [markdown]
# Finally, the following code cell plots the potential energy surface.


# %%
def plot_pes():
    plt.close("pes")
    fig, ax = plt.subplots(num="pes")
    xs = np.linspace(-30, 30, 201)
    ys = np.linspace(-20, 20, 201)
    coords = np.array(np.meshgrid(xs, ys)).transpose(1, 2, 0)
    energies = potential_energy_force(coords)[0]
    cf = ax.contourf(xs, ys, energies / EV, levels=20)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [a$_0$]")
    ax.set_ylabel("y [a$_0$]")
    ax.set_title("Potential Energy Surface")
    fig.colorbar(cf, ax=ax, label="Energy [eV]")


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

    If block_size is larger than 1,
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


def integrate(coords: ArrayLike, vels: ArrayLike, nstep: int, block_size: int = 1):
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
    block_size
        The block_size with which to record the trajectory data.

    Returns
    -------
    trajectory
        A Trajectory object holding all the results.
    """
    traj = Trajectory.empty(coords.shape[:-1], nstep // block_size, TIMESTEP * block_size)
    energies, forces = potential_energy_force(coords)
    delta_vels = forces * (0.5 * TIMESTEP / MASS)

    vels_block = 0
    for istep in range(traj.nstep * block_size):
        vels += delta_vels
        coords += vels * TIMESTEP
        energies, forces = potential_energy_force(coords)
        delta_vels = forces * (0.5 * TIMESTEP / MASS)
        vels += delta_vels
        vels_block += vels
        if istep % block_size == block_size - 1:
            itraj = istep // block_size
            traj.coords[itraj] = coords
            traj.vels[itraj] = vels_block / block_size
            traj.potential_energies[itraj] = energies
            traj.kinetic_energies[itraj] = (0.5 * MASS) * (vels**2).sum(axis=-1)
            vels_block = 0
    return traj


# %% [markdown]
# As a quick test, the following code cell integrates the equations of motion
# for a single particle with a small initial velocity.
# In this case, the particle oscillates around the origin
# and one can easily verify that the total energy is conserved.


# %%
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
    ax.legend()


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
    vels = np.array([[1e-3, 1e-4], [1.000001e-3, 1e-4]])
    traj = integrate(np.zeros((2, 2)), vels, 1500)
    plt.close("chaos")
    _, ax = plt.subplots(num="chaos")
    ax.plot([0], [0], "o", color="k", label="Initial position")
    ax.plot(traj.coords[:, 0, 0], traj.coords[:, 0, 1], color="C1", label="Trajectory 1")
    ax.plot(
        traj.coords[:, 1, 0],
        traj.coords[:, 1, 1],
        color="C3",
        ls=":",
        label="Trajectory 2",
    )
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [a$_0$]")
    ax.set_ylabel("y [a$_0$]")
    ax.legend()
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
#
# Note that the random walk is only found in a specific energy window.
# If the energy is too small,
# the particles will oscillate around a local potential energy minimum.
# If the energy is too large, or just high enough to cross barriers,
# the particles will follow almost linear paths over the surface.

# %% [markdown]
# ## Surface diffusion without block averages
#
# This section considers 100 independent particles whose initial velocities
# have the same magnitude but whose directions are random.
# The time-dependent particle velocities are used as inputs for STACIE
# to compute the diffusion coefficient.


# %%
def demo_stacie(block_size: int = 1):
    """Simulate particles on a surface and compute the diffusion coefficient.

    Parameters
    ----------
    block_size
        The block size for the block averages.
        If 1, no block averages are used.

    Returns
    -------
    result
        The result of the STACIE analysis.
    """
    natom = 100
    nstep = 20000
    rng = np.random.default_rng(42)
    vels = rng.normal(0, 1, (natom, 2))
    vels *= 9.7e-4 / np.linalg.norm(vels, axis=1).reshape(-1, 1)
    traj = integrate(np.zeros((natom, 2)), vels, nstep, block_size)

    plt.close(f"trajs_{block_size}")
    _, ax = plt.subplots(num=f"trajs_{block_size}", figsize=(6, 6))
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

    # Define units and conversion factors used for screen output and plotting.
    # This does not affect numerical values stored in the result object.
    uc = UnitConfig(
        acint_symbol="D",
        acint_unit=sc.value("atomic unit of time")
        / sc.value("atomic unit of length") ** 2,
        acint_unit_str="m$^2$/s",
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
    # the Lorentz model is not designed to handle.
    result = estimate_acint(
        spectrum, LorentzModel(), fcut_max=TERAHERTZ, verbose=True, uc=uc
    )

    # Plotting
    plt.close(f"spectrum_{block_size}")
    _, ax = plt.subplots(num=f"spectrum_{block_size}")
    plot_fitted_spectrum(ax, uc, result)
    plt.close(f"extras_{block_size}")
    _, axs = plt.subplots(2, 2, num=f"extras_{block_size}")
    plot_extras(axs, uc, result)
    return result


# %%
result_1 = demo_stacie()

# %% [markdown]
# The spectrum has several peaks related to oscillations of the particles
# around a local minimum.
# These peaks are irrelevant to the diffusion coefficient.
# The broad peak at zero frequency is used by STACIE
# to derive the diffusion coefficient.
# The obtained value is not directly comparable to experiment
# because the 2D lattice model for the surface
# is not based on an experimental case.
# However, the order of magnitude is comparable to the self-diffusion constants
# of pure liquids {cite:p}`baba_2022_prediction`.
#
# It is also interesting to compare the integrated and exponential autocorrelation time,
# as they are not the same in this case.

# %%
print(f"corrtime_exp = {result_1.corrtime_exp / PICOSECOND:.3f} ps")
print(f"corrtime_int = {result_1.corrtime_int / PICOSECOND:.3f} ps")

# %% [markdown]
# The integrated autocorrelation time is smaller than the exponential one
# because the former is an average of all time scales of the particle velocities.
# This includes the slow diffusion and faster oscillations in local minima.
# In contrast, the exponential autocorrelation time only represents the slow diffusive motion.
#
# Finally, it is well known that the velocity autocorrelation function
# of molecules in a liquid decays according to a power law {cite:p}`alder_1970_decay`.
# One might wonder why the Lorentz model can be used here since it implies that diffusion
# can be described with an exponentially decaying autocorrelation function.
# The system in this notebook exhibits exponential decay
# because every particle only interacts with the surface,
# and not with each other, such that there are no collective modes with long memory effects.

# %% [markdown]
# ## Surface diffusion with block averages
#
# This section repeats the same example,
# but now with block averages of velocities.
# [Block averages](../preparing_inputs/block_averages.md)
# are primarily useful for reducing storage requirements
# when saving trajectories to disk before processing them with STACIE.
# In this example, the block size is determined by the following guideline:

# %%
print(np.pi * result_1.corrtime_exp / (10 * TIMESTEP))

# %% [markdown]
# Let's use a block size of 60 to stay on the safe side.

# %%
result_60 = demo_stacie(60)

# %% [markdown]
# As expected, there are no significant changes in the results.
#
# It is again interesting to compare the integrated and exponential autocorrelation times.

# %%
print(f"corrtime_exp = {result_60.corrtime_exp / PICOSECOND:.3f} ps")
print(f"corrtime_int = {result_60.corrtime_int / PICOSECOND:.3f} ps")

# %% [markdown]
# The exponential autocorrelation time is unaffected by the block averages.
# However, the integrated autocorrelation time has increased
# and is now closer to the exponential value.
# Taking block averages removes the fastest oscillations,
# causing the integrated autocorrelation time to be dominated by slow diffusive motion.

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
acint_unit = sc.value("atomic unit of time") / sc.value("atomic unit of length") ** 2
acint_1 = result_1.acint / acint_unit
if abs(acint_1 - 5.80e-7) > 5e-9:
    raise ValueError(f"Wrong acint (no block average): {acint_1:.2e}")
acint_60 = result_60.acint / acint_unit
if abs(acint_60 - 5.80e-7) > 5e-9:
    raise ValueError(f"Wrong acint (block size 60): {acint_60:.2e}")
