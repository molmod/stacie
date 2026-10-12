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
# The initial production runs consist of 800 blocks, i.e. 8000 steps (40 ps) of simulation time.
#
# The spectrum of the charge current has a minimum at zero frequency,
# where its value is the ionic conductivity.
# At higher frequencies, the spectrum rises,
# because the ions are temporarily confined by their neighbors and oscillate in place,
# while on longer timescales they escape and diffuse.
# The [Debye model](#section-debye-target) describes the onset of this rise
# and is used throughout this notebook.
# It has $P=3$ parameters.
# Only the amount of data differs between the analyses below:
# the same 100 trajectories are analyzed after one, two and three simulation stages.
#
# In total 100 NVE production runs were performed.
# For each run, the system was first equilibrated in the NVT and later NPT ensemble.
# According to the section [](../preparing_inputs/data_sufficiency.md),
# 100 runs should be sufficient to obtain a relative error on the ionic conductivity of about 1%:
#
# $$ \epsilon_\text{rel} \approx \frac{1}{\sqrt{20 P M}} \approx 0.0075$$
#
# where $P$ is the number of parameters in the model
# and $M = 100 \times 3$ is the number of independent input sequences.
# (100 trajectories, 3 Cartesian components of the charge current per trajectory)
# The guideline of $400 P$ blocks would call for 1200 blocks with this model,
# somewhat more than the 800 blocks of the initial production runs.
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
    DebyeModel,
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
# The `analyze` function applies the same analysis with STACIE
# to different amounts of data (initial and extended production runs).
# After the analysis, it generates screen output and figures
# as discussed in the [minimal example](minimal.py).

# %%
BOLTZMANN_CONSTANT = sc.value("Boltzmann constant")  # J/K


def analyze(npart: int = 1, ntraj: int = 100, neff_max: int | None = 1000) -> float:
    """Analyze MD trajectories to compute the ionic conductivity.

    Parameters
    ----------
    npart
        The number of parts in the simulation to load.
        The default value of 1 corresponds to only loading the initial production runs.
    ntraj
        The number of trajectories to load.
    neff_max
        The maximum number of effective points in the fit, see :func:`stacie.estimate_acint`.

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
    result = estimate_acint(
        spectrum, DebyeModel(), neff_max=neff_max, verbose=True, uc=uc
    )

    # Plot some basic analysis figures.
    prefix = "conductivity"
    plt.close(f"{prefix}_fitted")
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
conductivity_1 = analyze()

# %% [markdown]
# The Debye model describes the spectrum well up to a cutoff frequency of about 2 THz.
# The fit uses 83 effective points, which is sufficient for a three-parameter model.
# The relative error on the conductivity is 1.4%,
# which is of the same order of magnitude as the coarse estimate of 0.75%.
#
# The Debye model also provides a relaxation time of about 94 fs.
# Its relative error (12%) is much larger than that of the conductivity (1.4%),
# so the curvature of the dip is only moderately constrained by the initial runs.
# The cutoff weight in the extra plots is concentrated around 2 THz
# and drops to zero well before the lowest and highest cutoff frequencies are reached,
# so the cutoff scan is not limited by the grid.
# A finer frequency grid is still useful to constrain the model more tightly.
# Hence, we extended the production runs by 8000 additional steps,
# of which the results are discussed in the following subsection.

# %% [markdown]
# ## Analysis of the Extended Production Simulation (8000 + 8000 steps)
#
# We simply call the same `analyze()` function, but now with `npart=2`,
# which loads the initial production runs and their first extension.

# %%
conductivity_2 = analyze(npart=2)

# %% [markdown]
# With twice as much data, the number of effective points in the fit also roughly doubles (157),
# and the uncertainty on the conductivity decreases from 4.9 to 3.5 S/m.
# The result is consistent with the one from the initial runs.
# The cutoff criterion Z-score is slightly above the ideal limit of 2,
# but this does not affect the conclusions,
# because the conductivity hardly changes over the cutoff scan (extra plot at the top right).
# To obtain more robust results,
# we extended the simulations once more.
# We added 184000 more steps, resulting in a simulation time of 1 ns for each of the 100 trajectories.

# %% [markdown]
# ## Analysis of the Extended Production Simulation (8000 + 8000 + 184000 steps)
#
# We simply call the same `analyze()` function, but now with `npart=3`,
# which loads the initial production runs and their first and second extensions.

# %%
conductivity_3 = analyze(npart=3, neff_max=4000)

# %% [markdown]
# The full production runs give a very detailed spectrum.
# With the default upper limit of 1000 effective points (`neff_max`),
# the cutoff scan is stopped by this limit rather than by the data.
# Hence, the call above raises `neff_max` to 4000,
# which allows the model to be fitted to higher frequencies.
# The scan now ends by itself, at about 2400 effective points,
# because the cutoff criterion rises well beyond its minimum.
# The average cutoff frequency is about 1.8 THz.
# The relative error estimate on the conductivity is 0.29%
# (with the default limit, it would be 0.33%).
# The relaxation time of about 92 fs agrees with the values from the shorter runs,
# and its relative error improves from 9% (default limit) to 3%.
# This is the main benefit of the higher limit:
# the conductivity is well determined because it is the value of the spectrum at zero frequency,
# while the relaxation time depends on the curvature of the dip,
# which is better constrained by the additional frequencies.
#
# The price is that the cutoff criterion Z-score rises to about 3,
# slightly above the ideal limit of 2, as for the extended runs above.
# This shows that the Debye model starts to deviate from the spectrum at the higher frequencies,
# which is expected because it only describes the onset of the resonance peak.
# The conductivity changes by less than one standard error when the limit is raised,
# so this deviation does not affect the conclusions.

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
# The table below lists the STACIE results discussed above and some results from the literature.
#
# | Ensemble | Simulated time [ns] | Density [g/cm<sup>3</sup>] | Conductivity [S/m] | Reference |
# | -------- | ------------------: | -------------------------: | -----------------: | --------- |
# | NpT+NVE  | 4                   | 1.454 ± 0.014              | 342.4 ± 4.9        | init debye |
# | NpT+NVE  | 8                   | 1.454 ± 0.014              | 345.8 ± 3.5        | ext1 debye |
# | NpT+NVE  | 100                 | 1.454 ± 0.014              | 350.7 ± 1.0        | ext2 debye |
# | NpT+NVT  | 6                   | 1.456                      | 348 ± 7            | {cite:p}`wang_2020_comparison` |
# | NpT+NVT  | > 5                 | 1.444                      | ≈ 310              | {cite:p}`wang_2014_molecular` |
# | Experiment | N.A.              | 1.542 ± 0.006              | 366 ± 3            | {cite:p}`janz_1968_molten` {cite:p}`bockris_1961_self` |
#
# The comparison shows that the results obtained with STACIE align reasonably well with the literature.
# In terms of statistical efficiency,
# STACIE achieves comparable or smaller error bars for about the same simulation time.
# The deviation from experiment is attributed to the approximations in the NaCl potential.
# {cite:p}`wang_2020_comparison`
#
# Finally, this example shows how the quality of the result improves with the amount of data.
# With a model that suits the spectrum,
# the effective number of points in the fit grows with the simulation time.
# As a result, the uncertainty on the conductivity decreases steadily:
# from 4.9 S/m (4 ns) to 3.5 S/m (8 ns) and 1.0 S/m (100 ns).
# This follows the inverse square root scaling with simulation time,
# which is expected when the number of effective points grows in proportion to the simulation time.


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
if abs(conductivity_1 - 342) > 5:
    raise ValueError(f"wrong conductivity (production): {conductivity_1:.0f}")
if abs(conductivity_2 - 346) > 4:
    raise ValueError(f"wrong conductivity (production): {conductivity_2:.0f}")
if abs(conductivity_3 - 351) > 2:
    raise ValueError(f"wrong conductivity (production): {conductivity_3:.0f}")
if abs(density - 1.449) > 0.02:
    raise ValueError(f"wrong density (production): {density:.3f}")
