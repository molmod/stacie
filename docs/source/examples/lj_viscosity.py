#!/usr/bin/env python

# %% [markdown]
# # Viscosity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)
#
# This example shows how to calculate viscosity
# from pressure tensor data obtained via a LAMMPS simulation.
# The simulation consists of 1372 atoms
# and is conducted at the thermodynamic state $\rho^*=0.8442$ and $T^*=0.722$,
# which corresponds to a liquid phase near the triple point $(\rho^*=0.0845$ and $T^*=0.69)$.
# This liquid state is known to exhibit slow relaxation times,
# which complicates the convergence of transport properties and
# makes it a popular benchmark for computational methods.
#
# All LAMMPS simulation inputs can be found in the directory `docs/data/lammps_lj3d`
# in Stacie's source repository.
#
# The simulations are performed in two stages:
#
# 1. An exploratory simulation to equilibrate the system and to obtain a preliminary estimate of the viscosity
#    and the time scales of the anisotropic pressure fluctuations.
#    Details of this run can be found in the LAMMPS input file
#    [../../data/lammps_lj3d/exploration/in.lammps](../../data/lammps_lj3d/exploration/in.lammps).
# 2. A set of 100 production simulations to get a more accurate estimate of the viscosity,
#    with the block averaged trajectory data to reduce storage requirements.
#    Details of the production runs can be found the LAMMPS inputs
#    [../../data/lammps_lj3d/template-prod-init.lammps](../../data/lammps_lj3d/template-prod-init.lammps)
#    and [../../data/lammps_lj3d/template-prod-extend.lammps](../../data/lammps_lj3d/template-prod-extend.lammps).
#    These inputs are actually [Jinja2](https://jinja.palletsprojects.com/) templates
#    that are rendered with different random seeds for each run.
#    The production was initially planned for 4000 steps, but it was extended by 4000 more steps.
#
# In addition to demonstrating how to use Stacie,
# this notebook also illustrates its efficient use of the trajectory information.
# The results in this notebook are comparable in statistical uncertainty to state-of-the-art results,
# despite using only a small fraction of the data as input.
#
# The LAMMPS input files contain commands to write output files
# that can be directly loaded with NumPy without any additional converters or wrappers.
# The following output files are used in this notebook:
#
# - In `docs/data/lammps_lj3d/exploration`:
#     - `info.yaml`: simulation parameters that may be useful for post-processing
#     - `nve_thermo.txt`: instantaneous temperature and related quantities
#     - `nve_pressure.txt`: (off)diagonal instantaneous pressure tensor components
# - In `docs/data/lammps_lj3d/production/*`:
#     - `info.yaml`: simulation parameters that may be useful for post-processing.
#     - `nve_thermo.txt`: subsampled instantaneous temperature and related quantities
#     - `nve_pressure_blav.txt`: block-averaged (off)diagonal pressure tensor components
#
# All MD simulations and this notebook use reduced Lennard-Jones units.
# For convenience, the reduced unit of viscosity is denoted as η\*,
# and the reduced unit of time as τ\*.


# %%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from path import Path
from yaml import safe_load
from stacie import UnitConfig, compute_spectrum, estimate_acint, summarize_results
from stacie.plot import plot_fitted_spectrum, plot_criterion

# %%
mpl.rc_file("matplotlibrc")

# %% [markdown]
# ## Analysis of the Exploratory Simulation
# This simulation consists of two NVT equilibration runs:
# one to melt the initial FCC lattice and one to reach the desired temperature.
# It is followed by an NVE run of 50000 steps, which is used for post-processing.
#
# The following code cell defines several analysis functions employed in
# both exploratory and production simulations.
#
# - `get_indep_pcomps` transforms the pressure tensor components
#   into five independent off-diagonal contributions,
#   as explained in the [shear viscosity](../theory/properties/shear_viscosity.md) theory section.
# - `plot_temperature` plots the instantaneous temperature and compares it to the desired temperature.
# - `estimate_viscosity` calculates the viscosity and plots the results.
#   It also prints recommendations for data reduction (block averaging) and simulation time,
#   as explained in the following two sections of the documentation:
#     - [Autocorrelation Time](../theory/properties/autocorrelation_time.md)
#     - [Block Averages](../theory/advanced_topics/block_averages.md)


# %%


def get_indep_pcomps(pcomps):
    return np.array(
        [
            (pcomps[0] - 0.5 * pcomps[1] - 0.5 * pcomps[2]) / np.sqrt(3),
            0.5 * pcomps[1] - 0.5 * pcomps[2],
            pcomps[3],
            pcomps[4],
            pcomps[5],
        ]
    )


def plot_temperature(name, time, temperatures, de_temperature):
    av_temperature = np.mean(temperatures)
    plt.close(f"{name}_temp")
    _, ax = plt.subplots(num=f"{name}_temp")
    for temperature in temperatures:
        ax.plot(time, temperature, alpha=0.5, label="__nolegend__")
    ax.axhline(de_temperature, color="black", label=f"Desired: {de_temperature:.3f}")
    ax.axhline(
        av_temperature, color="black", ls=":", label=f"Average {av_temperature:.3f}"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature")
    ax.legend()
    return av_temperature


def estimate_viscosity(name, pcomps, av_temperature, volume, timestep):
    # Create the spectrum of the pressure fluctuations.
    # Note that the Boltzmann constant is 1 in reduced LJ units.
    uc = UnitConfig(
        acint_fmt=".3f",
        acint_unit_str="η*",
        freq_unit_str="1/τ*",
        time_fmt=".3f",
        time_unit_str="τ*",
    )
    spectrum = compute_spectrum(
        pcomps,
        prefactor=0.5 * volume / av_temperature,
        timestep=timestep,
    )

    # Estimate the viscosity from the spectrum.
    result = estimate_acint(spectrum, verbose=True)

    # Plot some basic analysis figures.
    plt.close(f"{name}_criterion")
    _, ax = plt.subplots(num=f"{name}_criterion")
    plot_criterion(ax, uc, result)
    plt.close(f"{name}_spectrum")
    _, ax = plt.subplots(num=f"{name}_spectrum")
    plot_fitted_spectrum(ax, uc, result)

    # Print the recommended block size and simulation time.
    print()
    print(summarize_results(result, uc))

    # Return the viscosity
    return result.acint


# %% [markdown]
# The next cell performs the analysis of the exploratory simulation.
# It prints the recommended block size and the simulation time for the production runs,
# and then generates the following figures:
#
# - The time-dependence of the instantaneous temperature.
# - The underfitting criterion to find the low-frequency part of the spectrum for parameter fitting.
# - The spectrum of the off-diagonal pressure fluctuations, and the model fitted to the spectrum.

# %%


def demo_exploration():
    lammps = Path("../../data/lammps_lj3d")

    # Load the configuration from the YAML file.
    name = "exploration"
    with open(lammps / f"{name}/info.yaml") as fh:
        info = safe_load(fh)

    # Load trajectory data.
    thermo = np.loadtxt(lammps / f"{name}/nve_thermo.txt").T
    pressure = np.loadtxt(lammps / f"{name}/nve_pressure.txt").T

    # Plot the instantaneous and desired temperature.
    time = thermo[0] * info["timestep"]
    av_temperature = plot_temperature(name, time, [thermo[1]], info["temperature"])

    # Compute the viscosity.
    pcomps = get_indep_pcomps(pressure[1:])
    return estimate_viscosity(
        name, pcomps, av_temperature, info["volume"], info["timestep"]
    )


eta_exploration = demo_exploration()

# %% [markdown]
# Two numbers are printed at the end (above the plots):
#
# - The recommended block size (in reduced time units)
#   for block averaging the pressure tensor components.
#   This corresponds to about 10 time steps,
#   which is used to write smaller files in the production simulations.
# - The recommended simulation time to be used for production runs,
#   corresponding to about 4000 steps.
#   This means that trajectories should be at least this long.
#   Although longer simulation can be performed to gather more statistics, lower uncertainties
#   can typically be obtained by running more independent simulations of 4000 steps.
#   Additionally, these shorter simulations can be run in parallel.
#   Even shorter simulations should be avoided,
#   as these will not sufficiently resolve the low-frequency part of the spectrum.
#
# A warning is also printed, by default, Stacie fits the spectrum using a  maximum of 1000 points
# (to limit the computational cost). Including more points would not provide useful information here,
# as the high-frequency limit of the spectrum is irrelevant for the viscosity.


# %% [markdown]
# ## Analysis of the Production Simulations
#
# We initially ran 100 independent production runs of 4000 steps.
# Each production run uses the last snapshot of the exploratory simulation as its initial state.
# This state is first re-equilibrated with an NVT run of 10000 steps.
# Then the simulation is continued in the NVE ensemble for 4000 steps, which is used for analysis.
# The relatively long equilibration phase ensures that production runs provide uncorrelated data.
# Note that there is no velocity rescaling after the NVT equilibration:
# each NVE run has a slightly different average temperature, which is intentional.
# This spread is necessary to make them (as a whole) representative for the NVT ensemble
# and to prevent any bias in the viscosity estimate.
#
# When performing the analysis only with the initial production runs,
# it becomes clear that the recommended simulation time from the exploration run was
# a bit biased towards the low side.
# By analyzing the initial production data, the recommended simulation time became about 8000 steps.
# We have therefore extended the production runs by an additional 4000 steps each.
#
# The following cell implements the analysis of the production simulations with 8000 steps.
# As you can see, the code is very similar to the analysis of the exploratory simulation.

# %%


def demo_production():
    lammps = Path("../../data/lammps_lj3d")

    # Load the configuration from the YAML file.
    with open(lammps / "exploration/info.yaml") as fh:
        info = safe_load(fh)
    name = "production"
    with open(lammps / f"{name}/replica_0000_part_00/info.yaml") as fh:
        info.update(safe_load(fh))

    # Load trajectory data, without hardcoding the number of runs and parts.
    # - `part_00` corresponds to the initial production run (4000 steps).
    # - `part_01` corresponds to the first extension of the production run (4000 extra steps).
    thermos = []
    pressures = []
    last_replica = -1
    for prod_dir in sorted(Path(lammps / name).glob("replica_*_part_*/")):
        replica = prod_dir.split("/")[-2].split("_")[1]
        run_thermo = np.loadtxt(prod_dir / "nve_thermo.txt")
        run_pressures = np.loadtxt(prod_dir / "nve_pressure_blav.txt")
        if replica == last_replica:
            thermos[-1].append(run_thermo)
            pressures[-1].append(run_pressures)
        else:
            # In case of thermo, which was created `fix print`, the first row
            # repeats the last one of the previous run.
            thermos.append([run_thermo[1:]])
            pressures.append([run_pressures])
            last_replica = replica
    thermos = [np.concatenate(parts).T for parts in thermos]
    pressures = [np.concatenate(parts).T for parts in pressures]

    # Plot the instantaneous and desired temperature.
    time = thermos[0][0] * info["timestep"]
    av_temperature = plot_temperature(
        name, time, [thermo[1] for thermo in thermos], info["temperature"]
    )

    # Compute the viscosity.
    pcomps = np.concatenate([get_indep_pcomps(pressure[1:]) for pressure in pressures])
    return estimate_viscosity(
        name,
        pcomps,
        av_temperature,
        info["volume"],
        info["timestep"] * info["block_size"],
    )


eta_production = demo_production()

# %% [markdown]
# ## Comparison to Literature Results
#
# Comprehensive literature surveys on computational estimates of the viscosity of a Lennard-Jones fluid
# can be found in {cite:p}`meier_2024_transport` and {cite:p}`viscardi_2007_transport1`.
# These papers also present new results, which are included in the table below.
# Since the simulation settings ($r_\text{cut}^{*}=2.5$, $N=1372$, $T^*=0.722$ and $\rho^{*}=0.8442$)
# are identical to those used in this notebook, the reported values should be directly comparable.
#
# | Method                     | Simulation time [τ\*] | Viscosity [η\*] | Reference |
# |----------------------------|-----------------------|-----------------|-----------|
# | EMD NVE (Helfand-Einstein) | 75000                 | 3.277 ± 0.098   | {cite:p}`meier_2024_transport` |
# | EMD NVE (Helfand-moment)   | 600000                | 3.268 ± 0.055   | {cite:p}`viscardi_2007_transport1` |
# | EMD NVE (Stacie)           | 2400                  | 3.236 ± 0.078   | This notebook |
#
# This comparison confirms that Stacie can reproduce a well-known viscosity result,
# and that it achieves a small statistical uncertainty with far less data than existing methods.
#
# To be fair, the simulation time only accounts for production runs.
# Our setup also includes a small exploration run (210 τ*)
# and a significant amount of equilibration runs (3000 τ*)
# to ensure that different production runs are uncorrelated.
# Even when these additional runs are included, the overall simulation time
# remains significantly lower than in the cited papers.

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(eta_exploration - 3.1) > 0.2:
    raise ValueError(f"wrong viscosity (exploration): {eta_exploration:.3e}")
if abs(eta_production - 3.23) > 0.1:
    raise ValueError(f"wrong viscosity (production): {eta_production:.3e}")
