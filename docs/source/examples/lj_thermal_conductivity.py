#!/usr/bin/env python

# %% [markdown]
# # Thermal Conductivity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)

# This example shows how to derive the thermal conductivity
# using heat flux data from a LAMMPS simulation.
# To keep the example simple, the production simulations from the
# [Viscosity of the Lennard-Jones liquid](lj_shear_viscosity.py) notebook are reused.
# The exploration phase is skipped and it is only checked *a posteriori* that the simulations
# contain sufficient data.
#
# This notebook does not only show how to use STACIE.
# It also illustrates how it makes effective use of trajectory information.
# The results in this notebook are comparable in terms of statistical uncertainty to
# state-of-the-art results, even though the simulation time is much shorter.
#
# The following output files from the LAMMPS simulations are used in this notebook:
#
# - In `docs/data/lammps_lj3d/production/*`
#     - `info.yaml`: simulation parameters that may be useful for post-processing
#     - `nve_thermo.txt`: subsampled instantaneous temperature and related quantities
#     - `nve_heatflux_blav.txt`: $x$, $y$, and $z$ components of the full heat flux vector.
#        (i.e. $J_x$, $J_y$, and $J_z$)
#
# All MD simulations and this notebook use reduced Lennard-Jones units.
# For convenience, the reduced unit of thermal conductivity is denoted as κ\*,
# and the reduced unit of time as τ\*.

# %% [markdown]
# %% [markdown]
# ::: {warning}
# A Lennard-Jones system only exhibits pairwise interactions,
# for which the LAMMPS command `compute/heat flux` produces valid results.
# For systems with three- or higher-body interactions, one cannot simply use the same command.
# Consult the theory section on [thermal conductivity](../theory/properties/thermal_conductivity.md)
# for more background.
# :::

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from path import Path
from yaml import safe_load
from stacie import UnitConfig, compute_spectrum, estimate_acint, ExpTailModel
from stacie.plot import plot_fitted_spectrum, plot_extras

# %%
mpl.rc_file("matplotlibrc")

# %% [markdown]
# ## Analysis of the Production Simulations
#
# The function `estimate_thermal_conductivity` implements the analysis,
# assuming the data have been read from the LAMMPS outputs and are passed as function arguments.


# %%
def estimate_thermal_conductivity(name, jcomps, av_temperature, volume, timestep):
    # Create the spectrum of the heat flux fluctuations.
    # Note that the Boltzmann constant is 1 in reduced LJ units.
    uc = UnitConfig(
        acint_fmt=".3f",
        acint_symbol="κ",
        acint_unit_str="κ*",
        freq_unit_str="1/τ*",
        time_fmt=".3f",
        time_unit_str="τ*",
    )
    spectrum = compute_spectrum(
        jcomps,
        prefactors=1 / (volume * av_temperature**2),
        timestep=timestep,
    )

    # Estimate the viscosity from the spectrum.
    result = estimate_acint(spectrum, ExpTailModel(), verbose=True, uc=uc)

    # Plot some basic analysis figures.
    plt.close(f"{name}_spectrum")
    _, ax = plt.subplots(num=f"{name}_spectrum")
    plot_fitted_spectrum(ax, uc, result)
    plt.close(f"{name}_extras")
    _, axs = plt.subplots(2, 2, num=f"{name}_extras")
    plot_extras(axs, uc, result)

    # Return the viscosity
    return result.acint


# %% [markdown]
# The following cell implements the analysis of the production simulations.


# %%
def demo_production():
    lammps = Path("../../data/lammps_lj3d")

    # Load the configuration from the YAML file.
    with open(lammps / "exploration/info.yaml") as fh:
        info = safe_load(fh)
    name = "production"
    with open(lammps / f"{name}/replica_0000_part_00//info.yaml") as fh:
        info.update(safe_load(fh))

    # Load trajectory data, without hardcoding the number of runs and parts.
    # - `part_00` corresponds to the initial production run (4000 steps).
    # - `part_01` corresponds to the first extension of the production run (4000 extra steps).
    thermos = []
    heatfluxes = []
    last_replica = -1
    for prod_dir in sorted(Path(lammps / name).glob("replica_*_part_*/")):
        replica = prod_dir.split("/")[-2].split("_")[1]
        run_thermo = np.loadtxt(prod_dir / "nve_thermo.txt")
        run_heatfluxes = np.loadtxt(prod_dir / "nve_heatflux_blav.txt")
        if replica == last_replica:
            thermos[-1].append(run_thermo)
            heatfluxes[-1].append(run_heatfluxes)
        else:
            # In case of thermo, which was created `fix print`, the first row
            # repeats the last one of the previous run.
            thermos.append([run_thermo[1:]])
            heatfluxes.append([run_heatfluxes])
            last_replica = replica
    thermos = [np.concatenate(parts).T for parts in thermos]
    heatfluxes = [np.concatenate(parts).T for parts in heatfluxes]

    # Compute the average temperature.
    av_temperature = np.mean([thermo[1].mean() for thermo in thermos])

    # Compute the thermal conductivity.
    # Note the that last three components should not be used.
    jcomps = np.concatenate([heatflux[1:4] for heatflux in heatfluxes])
    return estimate_thermal_conductivity(
        name,
        jcomps,
        av_temperature,
        info["volume"],
        info["timestep"] * info["block_size"],
    )


alpha_production = demo_production()

# %% [markdown]
# Compared to the viscosity analysis, the timescales of the heat flux tensor fluctuations
# are about half as short.
# The same simulation lengths and block sizes are used as in the viscosity notebook
# for simplicity.
# This factor-of-2 difference should not have a significant impact on the results.

# %% [markdown]
# ## Comparison to Literature Results
#
# A detailed literature survey of computational estimates of the thermal conductivity
# of a Lennard-Jones fluid can be found in {cite:p}`viscardi_2007_transport2`.
# Viscardi also computes new estimates, one of which is included in the table below.
# This value can be directly comparable to the current notebook,
# because the settings are identical
# $r_\text{cut}^{*}=2.5$, $N=1372$, $T^*=0.722$ and $\rho^{*}=0.8442$.
#
# | Method                     | Simulation time  [τ\*] | Thermal conductivity [κ\*] | Reference |
# |----------------------------|------------------------|----------------------------|-----------|
# | EMD NVE (STACIE)           | 2400                   | 6.982 ± 0.082              | This notebook |
# | EMD NVE (Helfand-moment)   | 600000                 | 6.946 ± 0.12               | {cite:p}`viscardi_2007_transport2` |
#
# This small comparison confirms that STACIE can reproduce a well-known thermal conductivity result,
# with small error bars, while using much less trajectory data than existing methods.
#
# Note that the simulation time mentioned in the table only covers the production runs.
# Our setup also includes a short exploration run (210 τ\*)
# and a significant amount of equilibration runs (3000 τ\*)
# to ensure that different production runs are uncorrelated.
# Even when we include these runs, the total simulation time is still much lower
# than in the work of Viscardi *et al*.

# ::: {note}
# The results in this study were obtained using
# [LAMMPS version 19 Nov 2024](https://github.com/lammps/lammps/releases/tag/patch_19Nov2024).
# Note that minor differences may arise when using a different version of LAMMPS,
# or even the same version compiled with a different compiler.
# :::

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(alpha_production - 7.0) > 0.2:
    raise ValueError(f"wrong thermal conductivity (production): {alpha_production:.3e}")
