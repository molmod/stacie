# %% [markdown]
# # Ionic Electrical Conductivity of Molten Sodium Chloride at 1100 K (OpenMM)
#
# :::{warning}
# This example notebook is work in progress.
# There are still some issues with the MD results obtained with OpenMM,
# which are discussed in the notebook.
# :::
#
# This notebook shows how to post-process trajectories from OpenMM simulations
# to calculate the ionic electrical conductivity.
# The OpenMM trajectories are converted to NPZ files within the Jupyter Notebooks of the simulation,
# making the approach here easily adaptable to other codes or physical systems.
# All OpenMM simulation notebooks can be found in the directory `docs/data/openmm_salt`
# in STACIE's source repository.
# The required theoretical background is explained the
# [](../properties/electrical_conductivity.md) section.
#
# The MD simulations are performed using the Born-Huggins-Mayer-Tosi-Fumi potential,
# which is a popular choice for molten salts. {cite:p}`tosi_1964_ionic`
# This potential does not use mixing rules and it is not natively implemented in OpenMM,
# but it can be incorporated using the `CustomNonbondedForce` and some creativity,
# see `docs/data/openmm_salt/bhmft.py` in the Git repository.
# The molten salt was simulated with a 3D periodic box of 1728 ions (864 Na$^+$ and 864 Cl$^-$).
# The time step in all simulations was 5 fs.
#
# Following the [](../preparing_inputs/molecular_dynamics.md),
# an initial block size of 10 steps (50 fs) was used.
# Because there is little prior knowledge on the structure of the spectrum,
# the exponential polynomial model (ExpPoly) with degrees $S=\{0, 1\}$ was used initially,
# i.e. with $P=2$ parameters.
# As explained in the section on [block averages](../preparing_inputs/molecular_dynamics.md),
# $400 P$ blocks were collected in the initial production runs,
# amounting to 8000 steps (40 ps) of simulation time.
#
# In total 100 NVE production runs were performed.
# For each run, the system was first equilibrated in the NVT and later NPT ensemble.
# According to the section [](../preparing_inputs/data_sufficiency.md),
# 100 runs should be sufficient to obtain a relative error on the ionic conductivity of about 1%:
#
# $$ \epsilon_\text{rel} \approx \frac{1}{\sqrt{20 P M}} \approx 0.0091$$
#
# where $P$ is the number of parameters in the model
# and $M = 100 \times 3$ is the number of independent input sequences.
# (100 trajectories, 3 Cartesian components of the charge current per trajectory)
#
# :::{note}
# The results in this example were obtained using
# [OpenMM 8.2.0](https://github.com/openmm/openmm/releases/tag/8.2.0).
# Minor differences may arise when using a different version of OpenMM,
# or even the same version compiled with a different compiler.
# :::

# %% [markdown]
# ## Library Imports and Configuration

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from path import Path
import scipy.constants as sc
from scipy.stats import chi2
from stacie import (
    ExpPolyModel,
    PadeModel,
    UnitConfig,
    compute_spectrum,
    estimate_acint,
    plot_fitted_spectrum,
    plot_extras,
)
from utils import plot_instantaneous_percentiles, plot_cumulative_temperature_histogram

# %%
mpl.rc_file("matplotlibrc")
# %config InlineBackend.figure_formats = ["svg"]

# %%
# You normally do not need to change this path.
# It only needs to be overridden when building the documentation.
DATA_ROOT = Path(os.getenv("DATA_ROOT", "./")) / "openmm_salt/output/"

# %% [markdown]
# ## Analysis of the NpT Equilibration Runs
#
# To validate that the equilibration runs have reached to proper temperature distribution,
# the following cell implements a plot of the percentiles (over the 100 trajectories)
# of a thermodynamic quantity (temperature or volume).


# %%
def plot_openmm_percentiles(
    ensemble: str,
    field: str,
    unitstr: str,
    unit: float = 1,
    npart: int = 1,
    ntraj: int = 100,
    expected: None = None,
    ymin: float | None = None,
    ymax: float | None = None,
):
    """Plot the temperature of the NpT equilibration runs."""
    time = None
    natom = None
    sequences = []
    time = None
    for itraj in range(ntraj):
        row = []
        if itraj == 0:
            time = []
        for ipart in range(npart):
            path_npz = DATA_ROOT / f"sim{itraj:04d}_part{ipart:02d}_{ensemble}_traj.npz"
            if not path_npz.exists():
                print(f"File {path_npz} not found, skipping.")
                row = None
                break
            data = np.load(path_npz)
            natom = len(data["atnums"])
            if itraj == 0:
                time.append(data["time"])
            row.append(data[field])
        if row is None:
            continue
        if itraj == 0:
            time = np.concatenate(time)
        row = np.concatenate(row)
        sequences.append(row)
    sequences = np.array(sequences)

    percents = np.array([95, 80, 50, 20, 5])
    if field == "temperature":
        temp_d = 1100
        ndof = 3 * natom - 3
        expected = chi2.ppf(percents / 100, ndof) * temp_d / ndof
        ymin = chi2.ppf(0.01, ndof) * temp_d / ndof
        ymax = chi2.ppf(0.99, ndof) * temp_d / ndof
    else:
        expected = None
    time_unit = 1e-12
    num = f"percentiles_{field}_{ensemble}"
    plt.close(num)
    _, ax = plt.subplots(num=num)
    plot_instantaneous_percentiles(
        ax,
        time / time_unit,
        sequences / unit,
        percents,
        None if expected is None else expected / unit,
        ymin,
        ymax,
    )
    ax.set_title(f"{field.title()} percentiles during the {ensemble.upper()} run")
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel(f"{field.title()} [{unitstr}]")


# %% [markdown]
# The following cell plots the temperature percentiles for the NVT and NPT equilibration runs.

# %%
plot_openmm_percentiles("nvt", "temperature", "K")
plot_openmm_percentiles("npt", "temperature", "K")

# %% [markdown]
# The percentiles look good for the equilibration runs:
# they quickly reach their theoretical values (black dotted lines) and then fluctuate around them.
# The following cell plots the temperature percentiles for the initial NVE production runs.

# %%
plot_openmm_percentiles("nve", "temperature", "K")

# %% [markdown]
# This is clearly not the correct temperature distribution!
# There is a known problem with restart files in the NVE ensemble in OpenMM.
# Due to a bug, it tends to lower the temperature of the system.
# More details on this issue can be found here:
# <https://github.com/openmm/openmm/issues/4948>
#
# The following cell plots the percentiles of the volume (over the 100 trajectories),
# which we can only use to validate that the volume distribution converges.
# However, we cannot trivially compare these percentiles to an expected distribution.

# %%
plot_openmm_percentiles("npt", "volume", "nm$^3$", unit=1e-27, expected="last")

# %% [markdown]
# This plot is not completely satisfactory either, as it suggests that the volume
# fluctuations of the 100 runs exhibit synchronized fluctuations,
# while they should be independent.
# (They use different random seeds for their MC Barostat.)

# %% [markdown]
# ## Reusable Code for the Analysis of the Production Runs
#
# The `analyze` function takes a few parameters to apply the same analysis with STACIE
# to different inputs (initial and extended production runs).
# After the analysis, it generates screen output and figures
# as discussed in the [minimal example](minimal.py).

# %%
BOLTZMANN_CONSTANT = sc.value("Boltzmann constant")  # J/K


def analyze(model, npart: int = 1, ntraj: int = 100) -> float:
    """Analyze MD trajectories to compute the ionic conductivity.

    Parameters
    ----------
    model
        The model fitted to the spectrum.
    npart
        The number of parts in the simulation to load.
        The default value of 1 corresponds to only loading the initial production runs.

    Returns
    -------
    acint
        The estimated ionic conductivity, mainly used for regression testing.
    """
    # Get the time step from the first NPZ file.
    time = np.load(DATA_ROOT / "sim0000_part00_nve_traj.npz")["time"]
    timestep = time[1] - time[0]

    def iter_sequences():
        """A generator that only loads one MD trajectory at a time in memory."""
        for itraj in range(ntraj):
            paths_npz = [
                DATA_ROOT / f"sim{itraj:04d}_part{ipart:02d}_nve_traj.npz"
                for ipart in range(npart)
            ]
            if not all(path_npz.exists() for path_npz in paths_npz):
                print(f"Some of {paths_npz} not found, skipping.")
                continue
            dipole = []
            for path_npz in paths_npz:
                data = np.load(path_npz)
                dipole.append(data["dipole"])
            dipole = np.concatenate(dipole, axis=1)
            data = np.load(paths_npz[0])
            prefactor = 1.0 / (
                data["volume"][0] * data["temperature"].mean() * BOLTZMANN_CONSTANT
            )
            # The finite difference is equivalent to a block-averaged charge current.
            current = np.diff(dipole, axis=1) / timestep
            yield prefactor, current

    # Configure units for output
    uc = UnitConfig(
        acint_symbol=r"\sigma",
        acint_unit_str=r"S/m",
        acint_fmt=".1f",
        time_unit=1e-15,
        time_unit_str="fs",
        time_fmt=".3f",
        freq_unit=1e12,
        freq_unit_str="THz",
    )

    # Perform the analysis with STACIE
    spectrum = compute_spectrum(
        iter_sequences(),
        timestep=timestep,
        prefactors=None,
        include_zero_freq=False,
    )
    result = estimate_acint(spectrum, model, verbose=True, uc=uc)

    # Plot some basic analysis figures.
    prefix = "conductivity"
    plt.close(f"{prefix}_spectrum")
    _, ax = plt.subplots(num=f"{prefix}_fitted")
    plot_fitted_spectrum(ax, uc, result)
    plt.close(f"{prefix}_extras")
    _, axs = plt.subplots(2, 2, num=f"{prefix}_extras")
    plot_extras(axs, uc, result)

    # Return the ionic conductivity.
    return result.acint


# %% [markdown]
# ## Analysis of the Initial Production Simulation
#
# The following cell computes the ionic conductivity of the molten salt at 1100 K,
# from the initial production runs (8000 steps each).

# %%
conductivity_1_01 = analyze(ExpPolyModel([0, 1]))

# %% [markdown]
# The analysis of the initial production runs shows that the trajectories are not yet sufficient
# for a reliable interpretation of the autocorrelation integrals:
#
# - Only 15 effective points are used for the fitting.
# - The relative error is 3.1%, while higher than the coarse estimate of 0.9%,
#   it is of the right order of magnitude.
#
# The extra plots reveal another reason for extending the MD simulations.
# The cutoff weight is significant at the lowest cutoff frequency,
# suggesting that a finer grid with lower frequencies could reveal new details.
# Hence, we extended the production runs by 8000 additional steps to refine the frequency grid,
# of which the results are discussed in the following subsection.

# %% [markdown]
# ## Analysis of the Extended Production Simulation (8000 + 8000 steps)
#
# We simply call the same `analyze()` function, but now with `npart=2`,
# which loads the initial production runs and their first extension.

# %%
conductivity_2_01 = analyze(ExpPolyModel([0, 1]), npart=2)

# %% [markdown]
# The extended analysis shows that the results are starting to converge.
# However, the number of fitted points is still only 26, which is relatively low.
# To get more robust results, we extended the simulations once more.
# We added 184000 more steps, resulting in a simulation time of 1 ns for each of the 100 trajectories.

# %% [markdown]
# ## Analysis of the Extended Production Simulation (8000 + 8000 + 184000 steps)
#
# We simply call the same `analyze()` function, but now with `npart=3`,
# which loads the initial production runs and their first and second extensions.


# %%
conductivity_3_01 = analyze(ExpPolyModel([0, 1]), npart=3)

# %% [markdown]
# The analysis of the full extended production runs leads to a modest improvement.
# However, the utility of the first-order term of the model is questionable,
# given that the slope is nearly zero and could go either way
# according to the confidence intervals of the model (green dashed curves).
# Hence, we first test a constant (white noise) model to the first part of the spectrum:

# %%
conductivity_3_0 = analyze(ExpPolyModel([0]), npart=3)

# %% [markdown]
# Another model to consider is the Pade model, not because we expect the ACF to decay exponentially,
# but because it features well-behaved high-frequency limits, which can facilitate the regression.

# %%
conductivity_3_p = analyze(PadeModel([0, 2], [2]), npart=3)

# %% [markdown]
# This is indeed a successful regression, with 829 effective points for a three-parameter model.
# The relative error estimate on the final result is 0.37%.

# %% [markdown]
# ## Density
#
# To enable a proper comparison with the experimental and other simulation results,
# we also need to estimate the density of the system.
# This is done by averaging the density over the NpT trajectories from the production runs.


# %%
def estimate_density(ntraj: int = 100):
    densities = []
    molar_vols = []
    masses = {11: 22.990, 17: 35.45}  # g/mol
    avogadro = 6.02214076e23  # 1/mol
    for itraj in range(ntraj):
        path_npz = DATA_ROOT / f"sim{itraj:04d}_part00_npt_traj.npz"
        if not path_npz.exists():
            print(f"File {path_npz} not found, skipping.")
            continue
        data = np.load(path_npz)
        mass = sum(masses[atnum] for atnum in data["atnums"]) / avogadro
        volume = data["volume"] * 10**6  # from m³ to cm³
        densities.append(mass / volume)
        molar_vols.append(2 * avogadro * volume / len(data["atnums"]) / 2)
    density = np.mean(densities)
    print(f"Mass density: {density:.3f} ± {np.std(densities):.3f} g/cm³")
    print(f"Molar volume: {np.mean(molar_vols):.4f} ± {np.std(molar_vols):.4f} cm³/mol")
    return density


density = estimate_density()

# %% [markdown]
# ## Comparison to Literature Results
#
# Transport properties for this system are challenging to compute accurately.
# Consequently, simulation results from the literature may exhibit some variation.
# While the results should be broadly comparable to some extent, deviations may arise
# due to the differences in post-processing techniques,
# and the absence of reported error bars in some studies.
# Furthermore, in {cite:p}`wang_2014_molecular` smaller simulation cells were used
# (512 ions instead of 1728), which may also contribute to discrepancies.
#
# In the table below, we included some more results obtained with STACIE than those discussed above.
# We also computed the conductivity with the Pade model for all cases,
# which was a better choice in retrospect.
#
# | Ensemble | Simulated time [ns] | Density [g/cm<sup>3</sup>] | Conductivity [S/m] | Reference |
# | -------- | ------------------: | -------------------------: | -----------------: | --------- |
# | NpT+NVE  | 4                   | 1.454 ± 0.014              | 347 ± 10.9         | init expoly(0,1) |
# | NpT+NVE  | 8                   | 1.454 ± 0.014              | 354 ± 8.0          | ext1 expoly(0,1) |
# | NpT+NVE  | 100                 | 1.454 ± 0.014              | 353 ± 3.7          | ext2 expoly(0,1) |
# | NpT+NVE  | 100                 | 1.454 ± 0.014              | 353 ± 3.8          | ext2 expoly(0) |
# | NpT+NVE  | 4                   | 1.454 ± 0.014              | 343 ± 5.4          | init pade(0, 2; 2) |
# | NpT+NVE  | 8                   | 1.454 ± 0.014              | 346 ± 3.7          | ext1 pade(0, 2; 2) |
# | NpT+NVE  | 100                 | 1.454 ± 0.014              | 349 ± 1.3          | ext2 pade(0, 2; 2) |
# | NpT+NVT  | 6                   | 1.456                      | 348 ± 7    | {cite:p}`wang_2020_comparison` |
# | NpT+NVT  | > 5                 | 1.444                      | ≈ 310      | {cite:p}`wang_2014_molecular` |
# | Experiment | N.A.              | 1.542 ± 0.006              | 366 ± 3            | {cite:p}`janz_1968_molten` {cite:p}`bockris_1961_self` |
#
# The comparison shows that the results obtained with STACIE align reasonably well with the literature.
# In terms of statistical efficiency,
# STACIE achieves comparable or smaller error bars for about the same simulation time.
# The deviation from experiment is attributed to the approximations in the NaCl potential.
# {cite:p}`wang_2020_comparison`
#
# Finally, this example also shows why transport properties can be difficult to compute.
# As more data is collected, a more detailed spectrum is obtained.
# Simple models can struggle to explain the increasing amount of information.
# When extending the total simulation time from 8 ns to 100 ns,
# the effective number of points in the fit does not grow accordingly.
# As a result, the uncertainties decrease rather slowly with increasing simulation time.


# %% [markdown]
# ## Technical Details of the Analysis of the Literature Data
#
# References for the experimental data:
#
# - Density {cite:p}`janz_1968_molten`
# - Ionic conductivity {cite:p}`janz_1968_molten`

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
    molar_volume = molar_mass / density  # m³/mol
    conductivity = molar_conductivity / molar_volume
    conductivity_std = molar_conductivity_std / molar_volume
    print("Conductivity [S/m]", conductivity)
    print("Conductivity std [S/m]", conductivity_std)


convert_molar_conductivity()

# %% [markdown]
# ## Validation of the Production Runs
#
# To further establish that our NVE runs together represent the NpT ensemble,
# the following two cells perform additional validation checks.
#
# - A plot of the conserved quantity of the separate NVE runs, to detect any drift.
# - The distribution of the instantaneous temperature,
#   which should match the desired NpT distribution.
#   For each individual NVE run and for the combined NVE runs, cumulative distributions are plotted.
#   The function also plots the expected cumulative distribution of the NpT ensemble.


# %%
def plot_total_energy(npart: int = 3, ntraj: int = 100):
    time = None
    energies = []
    for itraj in range(ntraj):
        if itraj == 0:
            time = []
        energies_traj = []
        for ipart in range(npart):
            path_npz = DATA_ROOT / f"sim{itraj:04d}_part{ipart:02d}_nve_traj.npz"
            if not path_npz.exists():
                print(f"File {path_npz} not found, skipping.")
                continue
            data = np.load(path_npz)
            if itraj == 0:
                time.append(data["time"])
            energies_traj.append(data["total_energy"])
        if itraj == 0:
            time = np.concatenate(time)
        energies.append(np.concatenate(energies_traj))

    num = "total_energy"
    plt.close(num)
    _, ax = plt.subplots(num=num)
    for energies_traj in energies:
        plt.plot(time, energies_traj)
    plt.title("Total energy of the NVE production runs")
    plt.xlabel("Time [ps]")
    plt.ylabel("Total energy [kJ/mol]")


plot_total_energy()

# %% [markdown]
# There is no noticeable drift in the total energy of the NVE runs.
# Apart from the usual (and acceptable) numerical noise, the total energy is conserved perfectly.


# %%
def plot_temperature_production(npart: int = 3, ntraj: int = 100):
    """Plot cumulative distributions of the instantaneous temperature."""
    # Load the temperature data from the NVE production runs.
    natom = None
    temps = []
    for itraj in range(ntraj):
        temps.append([])
        for ipart in range(npart):
            path_npz = DATA_ROOT / f"sim{itraj:04d}_part{ipart:02d}_nve_traj.npz"
            if not path_npz.exists():
                print(f"File {path_npz} not found, skipping.")
                continue
            data = np.load(path_npz)
            natom = len(data["atnums"])
            temps[-1].append(data["temperature"])
    temps = np.array([np.concatenate(t) for t in temps])

    # Plot the instantaneous and desired temperature distribution.
    plt.close("tempprod")
    _, ax = plt.subplots(num="tempprod")
    ndof = 3 * natom - 3
    temp_d = 1100
    plot_cumulative_temperature_histogram(ax, temps, temp_d, ndof, "K")


plot_temperature_production()

# %% [markdown]
# Alas, as mentioned above, there is still a small mismatch between
# the obtained and expected NVT temperature distributions.
# This notebook will be updated after OpenMM issue [#4948](https://github.com/openmm/openmm/issues/4948)
# has been resolved.

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(conductivity_1_01 - 347) > 10:
    raise ValueError(f"wrong conductivity (production): {conductivity_1_01:.0f}")
if abs(conductivity_2_01 - 354) > 8:
    raise ValueError(f"wrong conductivity (production): {conductivity_2_01:.0f}")
if abs(conductivity_3_01 - 353) > 7:
    raise ValueError(f"wrong conductivity (production): {conductivity_3_01:.0f}")
if abs(conductivity_3_0 - 353) > 5:
    raise ValueError(f"wrong conductivity (production): {conductivity_3_0:.0f}")
if abs(conductivity_3_p - 349) > 3:
    raise ValueError(f"wrong conductivity (production): {conductivity_3_p:.0f}")
if abs(density - 1.449) > 0.02:
    raise ValueError(f"wrong density (production): {density:.3f}")
