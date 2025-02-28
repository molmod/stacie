#!/usr/bin/env python

# %% [markdown]
# # Viscosity of a Lennard-Jones Liquid Near the Tripple Point
#
# This example shows how to derive the viscosity
# from pressure tensor trajectories from a LAMMPS simulation.
# The simulated system consists of 1372 atoms
# and the thermodynamic state ($\rho^*=0.8442$ and $T^*=0.722$) corresponds to the liquid phase
# near the triple point ($\rho^*=0.0845$ and $T^*=0.69$).
# This liquid state is known to exhibit slow relaxation times,
# which troubles the convergence of transport properties,
# and which also makes it a popular choice for benchmarking computational methods.
#
# The simulations are performed in two stages:
#
# 1. An exploratory simulation to equilibrate the system and to get a first idea of the viscosity
#    and time scales of the anisotropic pressure fluctuations.
#    Details of this run can be found in the LAMMPS input
#    [../../data/lammps_lj3d/exploration/in.lammps](../../data/lammps_lj3d/exploration/in.lammps).
# 2. A set of 100 production simulations to get a more accurate estimate of the viscosity,
#    where the trajectory is block averaged to reduce storage requirements.
#    Details of the production runs can be found the LAMMPS input
#    [../../data/lammps_lj3d/template.lammps](../../data/lammps_lj3d/template.lammps).
#    This is actually a [Jinja2](https://jinja.palletsprojects.com/) template for the input,
#    which is rendered with a different random seed for each run.
#
# In addition to demonstrating how to use Stacie,
# this notebook also illustrates Stacie's efficienct use of the information in the trajectories.
# The results in this notebook are comparable in statistical uncertainty to state-of-the-art results,
# while using only a small fraction of the data as input.
#
# The LAMMPS input files contain commands to write output files
# that can be loaded directly with NumPy without any converters or wrappers.
# The following output files are used in this notebook:
#
# - In `docs/data/lammps_lj3d/exploration`:
#     - `info.yaml`: simulation parameters that may be useful for post-processing
#     - `nve_thermo.txt`: instantaneous temperature and related quantities
#     - `nve_pressure.txt`: (off)diagonal instantaneous pressure tensor components
# - In `docs/data/lammps_lj3d/production/????`:
#     - `info.yaml`: simulation parameters that may be useful for post-processing
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
# The following code cell defines some analysis functions that are used for
# both exploratory and production simulations:
#
# - `get_indep_pcomps` transforms the pressure tensor components
#   into five independent off-diagonal contributions,
#   as explained in the [shear viscosity](../theory/properties/shear_viscosity.md) theory section.
# - `plot_temperature` plots the instantaneous temperature and compares it to the desired temperature.
# - `estimate_viscosity` computes the viscosity and plots the results.
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
# The next cell implements the analysis of the exploratory simulation.
# It prints out the recommended block size and number of time steps for the extended simulation.
# It then produces the following figures:
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
#   This corresponds to about 10 time steps,
#   which is used to write smaller files in the production simulations.
# - The recommended simulation time to be used for production runs,
#   corresponding to about 10000 steps.
#   This means that trajectories should have at least this length.
#   If more statistics need to be collected, longer simulations can be performed,
#   but in practice lower uncertainties can be obtained by simply running
#   more independent simulations of 10000 steps.
#   It is also trivial to run these short simulations in parallel.
#   Even shorter simulations should be avoided,
#   as these will not sufficiently resolve the low-frequency part of the spectrum.
#
# A warning is printed that, by default, Stacie will fit the spectrum to a maximum of 1000 points
# (to limit the computational cost).
# While more points could be included, it will not provide useful information here
# since the high-frequency limit of the spectrum is irrelevant for the viscosity.


# %% [markdown]
# ## Analysis of the Production Simulations
#
# We ran 100 independent production runs of 10000 steps.
# Each production run uses the last snapshot of the exploratory simulation as its initial state.
# It first re-equilibrates the system with an NVT run of 10000 steps,
# and then performs the NVE run of the same length, which is used for post-processing.
# The equilibration phase is relatively long to ensure that all production runs
# provide uncorrelated data.
# Note that there is no velocity rescaling after the NVT equilibration:
# all NVE runs have a slightly different average temperature, which is intentional.
# This spread is necessary to make them (as a whole) representative for the NVT ensemble.
# Not doing so would bias the viscosity estimate.
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
# but the current production runs are already sufficient for accurate predictions.
# (The recommendations account for some uncertainty in the analysis of the exploration run.)

# %% [markdown]
# ## Comparison to Literature Results
#
# Detailed literature surveys of computational estimates of the viscosity of a Lennard-Jones fluid
# can be found in {cite:p}`meier_2024_transport` and {cite:p}`viscardi_2007_transport1`.
# These papers also present new results, which are included in the table below.
# These values should be directly comparable to the current notebook,
# because the settings are exactly the same:
# $r_\text{cut}^{*}=2.5$, $N=1372$, $T^*=0.722$ and $\rho^{*}=0.8442$.
#
# | Method                     | Simulation time<sup>\*</sup> | $\eta^*$       | Reference |
# |----------------------------|------------------------------|----------------|-----------|
# | EMD NVE (Helfand-Einstein) | 75000                        | 3.277 +- 0.098 | {cite:p}`meier_2024_transport` |
# | EMD NVE (Helfand-moment)   | 600000                       | 3.268 +- 0.055 | {cite:p}`viscardi_2007_transport1` |
# | EMD NVE (Stacie)           | 3000                         | 3.211 +- 0.064 | This notebook |
#
# This comparison confirms that Stacie can reproduce a well-known viscosity result,
# and that it obtains small error bars with much less trajectory data than existing methods.
#
# To be fair, the simulation time only accounts for production runs.
# Our setup also includes a small exploration run (210 τ<sup>\*</sup>)
# and a significant amount of equilibration (3000 τ<sup>\*</sup>)
# to ensure that different production runs are uncorrelated.
# Even when we include these runs, the total simulation time is still significantly lower
# than in the cited papers.

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
