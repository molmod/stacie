#!/usr/bin/env python3

# %% [markdown]
# # Ionic Conductivity and Self-diffusivity in Molten Sodium Chloride at 1100 K (OpenMM)
#
# This notebook shows how to post-process trajectories from OpenMM simulations
# to calculate ionic conductivity and self-diffusivity.
# The OpenMM trajectory is converted to NPZ files within the Jupyter Notebook of the simulation,
# making the approach here easily adaptable to other codes or physical systems.
#
# All OpenMM simulation inputs can be found in the directory `docs/data/openmm_salt`
# in Stacie's source repository.
#
# The MD simulations are performed using the Born-Huggins-Mayer-Tosi-Fumi potential,
# which is a popular choice for molten salts.
# This potential does not use mixing rules and it is not natively implemented in OpenMM,
# but it can be incorporated using the `CustomNonbondedForce` and some creativity,
# see `docs/data/openmm_salt/bhmft.py`.
#
# The simulations consist of two stages:
#
# 1. An **exploration** trajectory, where a system consisting of 1728 ions is equilibrated
#    in the NpT ensemble at 1100 K and 1 bar for 50 ps, followed by an NVE simulation for 50 ps,
#    which preserves the final instantaneous volume and energy of the NpT simulation.
#    This NVE trajectory provides an initial estimate of transport properties and helps to determine
#    a suitable simulation time and the block size for the production runs.
# 2. A set of 100 production simulations, where the final state of the equilibration run
#    is first re-equilibrated in the NpT ensemble for 50 ps, followed by 60 ps NVE run.
#    The NVE production trajectories are used for a more accurate estimate of
#    the ionic conductivity and self-diffusivity.

# %%
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from stacie import ExpPolyModel, UnitConfig, compute_spectrum, estimate_acint
from stacie.plot import plot_fitted_spectrum, plot_extras

# %%
mpl.rc_file("matplotlibrc")

# %% [markdown]
# ## Reusable Code for the Analysis
#
# The `analyze` function takes a few parameters to apply the same analysis with Stacie
# to different inputs, and with a different degree for the ExpPoly model.

# %%
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K


def analyze(paths_npz: list[str], transport_property: str, degrees: list[int]) -> float:
    """Analyze MD trajectories to compute the self-diffusivity or ionic conductivity.

    Parameters
    ----------
    paths_npz
        List of paths to the NPZ files containing the trajectory data.
    transport_property
        Specifies the transport property to compute.
        - 'na': Computes the self-diffusivity of sodium ions.
        - 'cl': Computes the self-diffusivity of chloride ions.
        - 'conductivity': Computes the ionic conductivity.
    degree
        The degree of the polynomial in the ExpPoly model fitted to the spectrum, by default 1.

    Returns
    -------
    acint
        The estimated transport property, mainly used for regression testing.
    """
    # Sanity check the NPZ files, may be empty due to globbing non-existing files.
    if len(paths_npz) == 0:
        raise ValueError("No NPZ files found in the input list.")

    # Get the time step from the first NPZ file.
    time = np.load(paths_npz[0])["time"]
    nstep = len(time)
    timestep = time[1] - time[0]

    def iter_sequences():
        """A generator that only loads one MD trajectory at a time in memory."""
        for path_npz in paths_npz:
            data = np.load(path_npz)
            atnums = data["atnums"]
            positions = data["atcoords"]
            if transport_property.lower() == "na":
                # Select only the positions sodium atoms for diffusion analysis
                positions = (
                    positions[:, atnums == 11].transpose(1, 2, 0).reshape(-1, nstep)
                )
                prefactor = 1.0
            elif transport_property.lower() == "cl":
                # Select only the positions of chlorine atoms for diffusion analysis
                positions = (
                    positions[:, atnums == 17].transpose(1, 2, 0).reshape(-1, nstep)
                )
                prefactor = 1.0
            elif transport_property.lower() == "conductivity":
                # Compute the instantaneous dipole moment
                # These are not really "positions", but we use the same variable name
                # to reuse the same code for the conductivity analysis.
                elementary_charge = 1.60217662e-19  # C
                positions = (
                    elementary_charge
                    * (
                        positions[:, atnums == 11].sum(axis=1)
                        - positions[:, atnums == 17].sum(axis=1)
                    ).T
                )
                prefactor = 1.0 / (
                    data["volume"] * data["temperature"] * BOLTZMANN_CONSTANT
                )
            else:
                raise ValueError(
                    f"Invalid transport_property: {transport_property}, "
                    "must be 'na', 'cl', or 'conductivity'"
                )

            # Construct a trajectory of "block-averaged" velocities,
            # to be used as input for the spectrum.
            # Note that the finite difference is just a block average
            # of velocities in the Verlet algorithm,
            # without additional approximations.
            velocities = np.diff(positions, axis=1) / timestep
            yield prefactor, velocities

    # Configure units for output
    uc = UnitConfig(
        acint_symbol=r"\sigma" if transport_property == "conductivity" else "D",
        acint_unit_str=r"S/m" if transport_property == "conductivity" else "m²/s",
        acint_fmt=".1f" if transport_property == "conductivity" else ".3e",
        time_unit=1e-12,
        time_unit_str="ps",
        time_fmt=".3f",
        freq_unit=1e12,
        freq_unit_str="THz",
    )

    # Perform the analysis with Stacie
    spectrum = compute_spectrum(
        iter_sequences(),
        timestep=timestep,
        prefactors=None,
        include_zero_freq=False,
    )
    result = estimate_acint(spectrum, ExpPolyModel(degrees), verbose=True, uc=uc)

    # Plot some basic analysis figures.
    prefix = f"{transport_property}_{len(paths_npz)}"
    plt.close(f"{prefix}_spectrum")
    _, ax = plt.subplots(num=f"{prefix}_fitted")
    plot_fitted_spectrum(ax, uc, result)
    plt.close(f"{prefix}_extras")
    _, axs = plt.subplots(2, 2, num=f"{prefix}_extras")
    plot_extras(axs, uc, result)

    # Return the diffusivity
    return result.acint


# %% [markdown]
# ## Analysis of the Exploratory Simulation
#
# The following cells compute the self-diffusivity of sodium and chloride ions,
# as well as the ionic conductivity of the molten salt at 1100 K.
# The results are preliminary, as they are based on a single, relatively short NVE trajectory.
# However, they provide insights into the relevant part of the spectrum, which helps to determine:
# - An appropriate simulation time to achieve sufficient frequency resolution.
# - A suitable block size to reduce storage by discarding irrelevant high-frequency data.
#
# Degree 1 is used to limit the number of unknowns in the fit because the data is limited.

# %%
path_nve_npz = "../../data/openmm_salt/output/exploration_nve_traj.npz"

# %%
analyze([path_nve_npz], "na", [0, 1])

# %%
analyze([path_nve_npz], "cl", [0, 1])

# %%
analyze([path_nve_npz], "conductivity", [0, 1])

# %% [markdown]
# As a compromise between the three recommendations, we choose a block size of 20 fs
# and a simulation time of 60 ps for the production trajectory.
# A total of 100 NVE trajectories were performed with these settings,
# but one may perform more simulations to improve the statistics.

# %% [markdown]
# ## Analysis of the Production Simulations
#
# The analysis of the production trajectories follows the same approach
# as the exploration trajectory, with two key differences:
# - The trajectory files are loaded as input.
# - The degree of the polynomial in the ExpPoly model is increased to 2.

# %%
paths_nve_npz = glob("../../data/openmm_salt/output/prod????_nve_traj.npz")

# %%
diffusivity_na = analyze(paths_nve_npz, "na", [0, 1, 2])

# %%
diffusivity_cl = analyze(paths_nve_npz, "cl", [0, 1, 2])

# %%
conductivity = analyze(paths_nve_npz, "conductivity", [0, 1, 2])

# %% [markdown]
#
# To enable a proper comparison with the experimental and other simulation results,
# we also need to estimate the density of the system.
# This is done by averaging the density over the NpT trajectories from the production runs.
#


# %%
def estimate_density(paths_npz: list[str]):
    densities = []
    molar_vols = []
    masses = {11: 22.990, 17: 35.45}  # g/mol
    avogadro = 6.02214076e23  # 1/mol
    for path_npz in paths_npz:
        data = np.load(path_npz)
        mass = sum(masses[atnum] for atnum in data["atnums"]) / avogadro
        volume = data["volume"] * 10**6  # cm³
        densities.append(mass / volume)
        molar_vols.append(2 * avogadro * volume / len(data["atnums"]) / 2)
    density = np.mean(densities)
    print(f"Mass density: {density:.3f} ± {np.std(densities):.3f} g/m³")
    print(f"Molar volume: {np.mean(molar_vols):.4f} ± {np.std(molar_vols):.4f} cm³/mol")
    return density


paths_npt_npz = glob("../../data/openmm_salt/output/prod????_npt_traj.npz")
density = estimate_density(paths_npt_npz)

# %% [markdown]
# ## Comparison to Literature Results
#
# Transport properties for this system are challenging to compute accurately,
# as reflected in the unusually shaped spectrum and MSD.
# Consequently, simulation results from the literature may exhibit some variation.
# While the results should be broadly comparable to some extent, deviations may arise
# due to the differences in post-processing techniques,
# and the absence of reported error bars in some studies.
# Furthermore, in {cite:p}`wang_2014_molecular` smaller simulation cells were used
# (512 ions instead of 1728), which may also contribute to discrepancies.
#
# | Method          | Simulated time [ps] | Density [g/cm<sup>3</sup>] | Na$^+$ diffusivity [10<sup>-9</sup>m<sup>2</sup>/s] | Cl$^-$ diffusivity [10<sup>-9</sup>m<sup>2</sup>/s] | Conductivity [S/m] | Reference |
# |-----------------|---------------------|----------------------------|-----------------------------------------------------|-----------------------------------------------------|--------------------|-----------|
# | NpT+NVE (BGMTF) | 6 ns                | 1.453 ± 0.004              | 8.354 ± 0.010                                       | 7.687 ± 0.009                                       | 340 ± 6            | This notebook |
# | NpT+NVT (BHMTF) | 1 ns (D), 6 ns (σ)  | 1.456                      | 8.8 ± 0.4                                           | 8.2 ± 0.5                                           | 348 ± 7            | {cite:p}`wang_2020_comparison` |
# | NpT+NVT (BHMTF) | > 5 ns              | 1.444                      | 9.36                                                | 8.14                                                | ≈ 310 (from plot)  | {cite:p}`wang_2014_molecular` |
# | Experiment      | N.A.                | 1.542 ± 0.006              | 9.0 ± 0.5                                           | 6.50 ± 0.14                                         | 366 ± 3            | {cite:p}`janz_1968_molten` {cite:p}`bockris_1961_self` |
#
# The comparison shows that the results obtained with Stacie align reasonably well with the literature.
# In terms of statistical efficiency, Stacie achieves comparable error bars.

# %% [markdown]
# ### Technical Details of the Analysis of the Literature Data
#
# References for the experimental data:
#
# - Density {cite:p}`janz_1968_molten`
# - Self-diffusion coefficients {cite:p}`bockris_1961_self`
# - Ionic conductivity {cite:p}`janz_1968_molten`
#
# The code below was used to compute the diffusion coefficients from the
# experimentally fitted Arrhenius equation in {cite:p}`bockris_1961_self`.


# %%
def compute_experimental_diffusivities():
    """Compute the diffusion coefficients from the experimental data."""
    # Parameters taken from Bockris 1961 (https://doi.org/10.1039/DF9613200218)
    molar_gas_constant = 1.98720425864083  # cal/(mol K)
    temperature = 1100  # K
    rt = molar_gas_constant * temperature

    # Sodium in NaCl
    a_na = 3.360e-7  # m²/s
    e_na = 7900  # cal/mol
    e_na_std = 110  # cal/mol
    exp_na = np.exp(-e_na / rt)
    print("D_na", a_na * exp_na)
    print("D_na_std", a_na * exp_na / rt * e_na_std)

    # Chloride in NaCl
    a_cl = 3.02e-7  # m²/s
    e_cl = 8390  # cal/mol
    e_cl_std = 40  # cal/mol
    exp_cl = np.exp(-e_cl / rt)
    print("D_cl", a_cl * exp_cl)
    print("D_cl_std", a_cl * exp_na / rt * e_cl_std)


compute_experimental_diffusivities()

# %% [markdown]
#
# The following cell converts a molar ionic conductivity from the literature back to a conductivity.


# %%
def convert_molar_conductivity():
    """Convert a specific conductance to a conductivity."""
    # Parameters taken from Wang 2020 (https://doi.org/10.1063/5.0023225)
    # and immediately converted to SI units
    molar_conductivity = 140 * 1e-4  # S m²/mol
    molar_conductivity_std = 3 * 1e-4  # S m²/mol
    density = 1.456 * 1e3  # kg/m³
    molar_mass = (22.990 + 35.45) * 1e-3  # kg/mol
    # avogadro = 6.02214076e23  # 1/mol
    molar_volume = molar_mass / density  # m³/mol
    conductivity = molar_conductivity / molar_volume
    conductivity_std = molar_conductivity_std / molar_volume
    print("Conductivity [S/m]", conductivity)
    print("Conductivity std [S/m]", conductivity_std)


convert_molar_conductivity()

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(diffusivity_na - 8.09e-9) > 2e-8:
    raise ValueError(f"wrong Na diffusivity (production): {diffusivity_na:.3e}")
if abs(diffusivity_cl - 7.32e-9) > 2e-8:
    raise ValueError(f"wrong Cl diffusivity (production): {diffusivity_cl:.3e}")
if abs(conductivity - 340) > 20:
    raise ValueError(f"wrong conductivity (production): {conductivity:.3e}")
if abs(density - 1.449) > 0.02:
    raise ValueError(f"wrong viscosity (production): {density:.3e}")
