#!/usr/bin/env python

# %% [markdown]
# # Thermal Conductivity of a Lennard-Jones Liquid Near the Triple Point

# This example shows how to derive the thermal conductivity
# using heat flux data from a LAMMPS simulation.
#
# This notebook shows how to use Stacie and how it makes effective use of trajectory information.
# The results in this notebook are comparable in terms of statistical uncertainty of the
# state-of-the-art results, even though the simulation time is much shorter.
#
# This notebook has the same structure is the one on
# [Viscosity for the Lennard-Jones liquid](lj_viscosity.py),
# and it post-processes outputs from the same LAMMPS simulations. However, it focuses only on
# the thermal conductivity and does not repeat the comments that were already in
# the viscosity notebook. It is assumed that you've worked through the viscosity notebook first.
#
# The following output files from the LAMMPS simulations are used in this notebook:
#
# - In `docs/data/lammps_lj3d/exploration`:
#     - `info.yaml`: simulation parameters that may be useful for post-processing
#     - `nve_thermo.txt`: instantaneous temperature and related quantities
#     - `nve_heatflux.txt`: diagonal elements of the heat flux tensor
# - In `docs/data/lammps_lj3d/production/0*`
#     - `info.yaml`: simulation parameters that may be useful for post-processing
#     - `nve_thermo.txt`: subsampled instantaneous temperature and related quantities
#     - `nve_heatflux_blav.txt`: $x$, $y$, and $z$ components of the full heat flux vector.
#        (i.e. $J_x$, $J_y$, and $J_z$)
#
# All MD simulations and this notebook use reduced Lennard-Jones units.
# For convenience, the reduced unit of thermal conductivity is written as κ<sup>\*</sup>,
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


# %%
def estimate_thermal_conductivity(name, jcomps, av_temperature, volume, timestep):
    # Create the spectrum of the heat flux fluctuations.
    # Note that the Boltzmann constant is 1 in reduced LJ units.
    uc = UnitConfig(
        acint_fmt=".3f",
        acint_symbol=r"\kappa",
        acint_unit_str=r"$\kappa^*$",
        freq_unit_str=r"1/$\tau^*$",
        time_unit_str=r"$\tau^*$",
    )
    spectrum = compute_spectrum(
        jcomps,
        prefactor=0.5 / (volume * av_temperature**2),
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

# %%


def demo_exploration():
    lammps = Path("../../data/lammps_lj3d")

    # Load the configuration from the YAML file.
    name = "exploration"
    with open(lammps / f"{name}/info.yaml") as fh:
        info = safe_load(fh)

    # Load trajectory data.
    thermo = np.loadtxt(lammps / f"{name}/nve_thermo.txt").T
    heatflux = np.loadtxt(lammps / f"{name}/nve_heatflux.txt").T

    # Plot the instantaneous and desired temperature.
    # time = thermo[0] * info["timestep"]
    # av_temperature = plot_temperature(name, time, [thermo[1]], info["temperature"])
    av_temperature = thermo[1].mean()

    # Compute the viscosity.
    return estimate_thermal_conductivity(
        name, heatflux[1:], av_temperature, info["volume"], info["timestep"]
    )


kappa_exploration = demo_exploration()
# %% [markdown]
# Compared to the viscosity analysis, the timescales of the heat flux tensor fluctuations
# are about half as short.
# The same simulation lengths and block sizes are used as in the viscosity notebook
# for simplicity.
# Our rather conservative guidelines work well even with this factor-of-2 difference.

# %% [markdown]
# ## Analysis of the Production Simulations
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
    heatfluxes = [
        np.loadtxt(path_txt).T
        for path_txt in glob(lammps / f"{name}/????/nve_heatflux_blav.txt")
    ]

    # Plot the instantaneous and desired temperature.
    # time = thermos[0][0] * info["timestep"]
    # av_temperature = plot_temperature(
    #    name, time, [thermo[1] for thermo in thermos], info["temperature"]
    # )
    av_temperature = np.mean([thermo[1].mean() for thermo in thermos])

    # Compute the viscosity, now splitting the trajectory into 500 blocks.
    # Note the that last three components should not be used.
    jcomps = np.concatenate([heatflux[1:4] for heatflux in heatfluxes])
    return estimate_thermal_conductivity(
        name,
        jcomps,
        av_temperature,
        info["volume"],
        info["timestep"] * info["block_size"],
    )


kappa_production = demo_production()

# %% [markdown]
# ## Comparison to Literature Results
#
# A detailed literature survey of computational estimates of the thermal conductivity
# of a Lennard-Jones fluid can be found in {cite:p}`viscardi_2007_transport2`.
# Viscardi also presents new results, one of them is included in the table below.
# This values can be directly comparable to the current notebook,
# because the settings are identical
# $r_\text{cut}^{*}=2.5$, $N=1372$, $T^*=0.722$ and $\rho^{*}=0.8442$.
#
# | Method                     | Simulation time<sup>\*</sup> | $\eta^*$       | Reference |
# |----------------------------|------------------------------|----------------|-----------|
# | EMD NVE (Helfand-moment)   | 600000                       | 6.946 +- 0.12  | {cite:p}`viscardi_2007_transport2` |
# | EMD NVE (Stacie)           | 3000                         | 7.041 +- 0.093 | This notebook |
#
# This comparison confirms that Stacie can reproduce a well-known thermal conductivity result,
# with smaller error bars, even using much less trajectory data than existing methods.
#
# Note that the simulation time mentioned in the table only covers the production runs.
# Our setup also includes a short exploration run (210 τ<sup>\*</sup>)
# and a long equilibration run (3000 τ<sup>\*</sup>)
# to ensure that different production runs are uncorrelated.
# Even when we include these runs, the total simulation time is still much lower
# than in the Helfland-moment paper.

# %% [markdown]
# :::{warning}
# LAMMPS `compute/heat flux` command is reported to produce unphysical results when many-body interaction terms (i.e. angle,
# dihedral, impropers..) are present {cite:p}`jamali_2019_octp`, {cite:p}`surblys_2019_application`, {cite:p}`boone_2019_heat`, {cite:p}`surblys_2021_methodology`.
# In those cases, one should use the `compute heat/flux` command with [`compute centroid/stress/atom` command](https://docs.lammps.org/compute_heat_flux.html).
# For systems with only 2-body interactions, using the `compute heat/flux` command with `compute stress/atom` command is not problematic.
# This warning is important for molecular systems.
# :::

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(kappa_exploration - 7.00) > 0.2:
    raise ValueError(f"wrong thermal conductivity (exploratory): {kappa_exploration:.3e}")
if abs(kappa_production - 7.04) > 0.1:
    raise ValueError(f"wrong thermal conductivity (extended): {kappa_production:.3e}")
