#!/usr/bin/env python

# %% [markdown]
# # Viscosity of a Lennard-Jones Liquid Near the Triple Point
#
# This example shows how to calculate viscosity
# from pressure tensor data obtained via a LAMMPS simulation.
# The simulation consists of 1372 atoms
# and is conducted at the thermodynamic state ($\rho^*=0.8442$ and $T^*=0.722$) which corresponds to the liquid phase
# near the triple point ($\rho^*=0.0845$ and $T^*=0.69$).
# This liquid state is known to exhibit slow relaxation times,
# which complicates the convergence of transport properties and
# makes it a popular benchmark for computational methods.
#
# The simulations are performed in two stages:
#
# 1. An exploratory simulation to equilibrate the system and to obtain a preliminary estimate of the viscosity
#    and the time scales of the anisotropic pressure fluctuations.
#    Details of this run can be found in the LAMMPS input file
#    [../../data/lammps_lj3d/exploration/in.lammps](../../data/lammps_lj3d/exploration/in.lammps).
# 2. A set of 100 production simulations to achieve a more accurate estimate of the viscosity,
#    with the block averaged trajectory data to reduce storage requirements.
#    Details of the production runs can be found the LAMMPS input file
#    [../../data/lammps_lj3d/template.lammps](../../data/lammps_lj3d/template.lammps).
#    Note that this input is actually a [Jinja2](https://jinja.palletsprojects.com/) template,
#    which is rendered with a different random seed for each run.
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
# - In `docs/data/lammps_lj3d/production/????`:
#     - `info.yaml`: simulation parameters that may be useful for post-processing.
#     - `nve_thermo.txt`: subsampled instantaneous temperature and related quantities
#     - `nve_pressure_blav.txt`: block-averaged (off)diagonal pressure tensor components
#
# All MD simulations and this notebook use reduced Lennard-Jones units.
# For convenience, the reduced unit of viscosity is written as η<sup>\*</sup>,
# and the reduced unit of time as τ<sup>\*</sup>.


# %%
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from path import Path
from yaml import safe_load
from stacie import UnitConfig, compute_spectrum, estimate_acint
from stacie.plot import plot_fitted_spectrum, plot_criterion

# %%
mpl.rc_file("matplotlibrc")

# %% [markdown]
# ## Analysis of the Exploratory Simulation
# This simulation consists of two NVT equilibration runs to melt the initial FCC lattice
# and to reach the desired temperature.
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
#   It also prints recommendations for data reduction (block averaging) and number of time steps,
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
        acint_unit_str=r"$\eta^*$",
        freq_unit_str=r"$1/\tau^*$",
        time_unit_str=r"$\tau^*$",
    )
    spectrum = compute_spectrum(
        pcomps,
        prefactor=0.5 * volume / av_temperature,
        timestep=timestep,
    )

    # Estimate the viscosity from the spectrum.
    result = estimate_acint(spectrum)

    # Plot some basic analysis figures.
    plt.close(f"{name}_criterion")
    _, ax = plt.subplots(num=f"{name}_criterion")
    plot_criterion(ax, uc, result)
    plt.close(f"{name}_spectrum")
    _, ax = plt.subplots(num=f"{name}_spectrum")
    plot_fitted_spectrum(ax, uc, result)

    # Give a recommendation for the block size
    block_size = (1 / 20) * result.corrtime_exp * np.pi
    print(f"Recommended block size: {block_size:.3f} τ*")

    # Give a recommendation for the simulation time
    num_steps = 40 * result.corrtime_exp * np.pi
    print(f"Recommended simulation time: {num_steps:.3f} τ*")

    # Return the viscosity
    return result.acint


# %% [markdown]
# The next cell performs the analysis of the exploratory simulation.
# It prints the recommended block size and the number of time steps for the extended simulation,
# and then generates the following figures:
#
# - The time-dependence of the instantaneous temperature.
# - The underfitting criterion to find the low-frequency part of the spectrum for parameter fitting.
# - The spectrum of the pressure fluctuations, and the model fitted to the spectrum.

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
#   This block size corresponds to roughly 10 time steps,
#   which is used to write smaller output files in the production simulations.
# - The recommended simulation time for production runs,
#   corresponding to approximately 10000 steps.
#   This means that trajectories should be at least this long.
#   Although longer simulation can be performed to gather more statistics, lower uncertainties
#   can typically be obtained by running more independent simulations of 10000 steps.
#   Additionally, these shorter simulations can be run in parallel.
#   Simulations shorter than this time should be avoided,
#   as these will not sufficiently resolve the low-frequency part of the spectrum.
#
# A warning is also printed, by default, Stacie fits the spectrum using a  maximum of 1000 points
# (to limit the computational cost). Including more points would not provide useful information here,
# as the high-frequency limit of the spectrum is irrelevant for the viscosity.


# %% [markdown]
# ## Analysis of the Production Simulations
#
# We ran 100 independent production runs of 10000 steps.
# Each run uses the final snapshot from the exploratory simulation as its initial state.
# First, the system is re-equilibrated with an NVT run of 10000 steps,
# followed by an NVE run of the same length for post-processing.
# The relatively long equilibration phase ensures that all production runs
# provide uncorrelated data.
# Note that there is no velocity rescaling after the NVT equilibration:
# each NVE runs have a slightly different average temperature, which is intentional.
# This spread is necessary to make them (as a whole) representative for the NVT ensemble
# and to prevent any bias in the viscosity estimate.
#
# The following cell implements the analysis of the production simulations.
# As you can see, the code is very similar to the analysis of the exploratory simulation.

# %%


def demo_production():
    lammps = Path("../../data/lammps_lj3d")

    # Load the configuration from the YAML file.
    with open(lammps / "exploration/info.yaml") as fh:
        info = safe_load(fh)
    name = "production"
    with open(lammps / f"{name}/0000/info.yaml") as fh:
        info.update(safe_load(fh))

    # Load trajectory data.
    thermos = [
        np.loadtxt(path_txt).T
        for path_txt in glob(lammps / f"{name}/????/nve_thermo.txt")
    ]
    pressures = [
        np.loadtxt(path_txt).T
        for path_txt in glob(lammps / f"{name}/????/nve_pressure_blav.txt")
    ]

    # Plot the instantaneous and desired temperature.
    time = thermos[0][0] * info["timestep"]
    av_temperature = plot_temperature(
        name, time, [thermo[1] for thermo in thermos], info["temperature"]
    )

    # Compute the viscosity, now splitting the trajectory into 500 blocks.
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
# Note that the recommended block size and simulation time of the final analysis have increased.
# One could rerun the simulations with these new parameters,
# but the current production runs already yield sufficiently accurate predictions.
# (The recommendations account for some uncertainty in the analysis of the exploration run.)

# %% [markdown]
# ## Comparison to Literature Results
#
# Comprehensive literature surveys on computational estimates of the viscosity of a Lennard-Jones fluid
# can be found in {cite:p}`meier_2024_transport` and {cite:p}`viscardi_2007_transport1`.
# These papers also present new results, which are included in the table below.
# Since the simulation settings ($r_\text{cut}^{*}=2.5$, $N=1372$, $T^*=0.722$ and $\rho^{*}=0.8442$)
# are identical to those used in this notebook, the reported values should be directly comparable.
#
# | Method                     | Simulation time<sup>\*</sup> | $\eta^*$       | Reference |
# |----------------------------|------------------------------|----------------|-----------|
# | EMD NVE (Helfand-Einstein) | 75000                        | 3.277 +- 0.098 | {cite:p}`meier_2024_transport` |
# | EMD NVE (Helfand-moment)   | 600000                       | 3.268 +- 0.055 | {cite:p}`viscardi_2007_transport1` |
# | EMD NVE (Stacie)           | 3000                         | 3.211 +- 0.064 | This notebook |
#
# This comparison confirms that Stacie can reproduce a well-known viscosity result,
# and achieves small error bars while using less trajectory data than the cited papers.
#
# Note that, the reported simulation time only accounts for production runs.
# Our setup also includes a short exploration run (210 τ<sup>\*</sup>)
# and a significantly long equilibration (3000 τ<sup>\*</sup>)
# to ensure that the different production runs are uncorrelated.
# Even when these additional runs are included, the overall simulation time
# remains significantly lower than that reported in the cited papers.

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(eta_exploration - 3.1) > 0.2:
    raise ValueError(f"wrong viscosity (exploratory): {eta_exploration:.3e}")
if abs(eta_production - 3.21) > 0.1:
    raise ValueError(f"wrong viscosity (extended): {eta_production:.3e}")
