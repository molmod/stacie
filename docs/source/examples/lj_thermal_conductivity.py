#!/usr/bin/env python

# %% [markdown]
# # Thermal Conductivity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)

# This example shows how to derive the thermal conductivity
# using heat flux data from a LAMMPS simulation.
# To keep the example simple, the production simulations from the
# [Viscosity of the Lennard-Jones liquid](lj_viscosity.py) notebook are reused.
# The exploration phase is skipped and it is only checked *a posteriori* that the simulations
# contain sufficient data.
#
# This notebook does not only show how to use Stacie.
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
        prefactor=0.5 / (volume * av_temperature**2),
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


kappa_production = demo_production()

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
# | EMD NVE (Helfand-moment)   | 600000                 | 6.946 ± 0.12               | {cite:p}`viscardi_2007_transport2` |
# | EMD NVE (Stacie)           | 2400                   | 7.058 ± 0.103              | This notebook |
#
# This small comparison confirms that Stacie can reproduce a well-known thermal conductivity result,
# with small error bars, while using much less trajectory data than existing methods.
#
# Note that the simulation time mentioned in the table only covers the production runs.
# Our setup also includes a short exploration run (210 τ\*)
# and a significant amount of equilibration runs (3000 τ\*)
# to ensure that different production runs are uncorrelated.
# Even when we include these runs, the total simulation time is still much lower
# than in the work of Viscardi *et al*.

# %% [markdown]
# ::: {warning}
#
#    The LAMMPS `compute/heat flux` command is reported to produce unphysical results
#    when many-body interactions (e.g. angle, dihedral, impropers) are present
#    {cite:p}`jamali_2019_octp`, {cite:p}`surblys_2019_application`,
#    {cite:p}`boone_2019_heat`, {cite:p}`surblys_2021_methodology`.
#    In that case, one should use the `compute heat/flux` command with
#    [`compute centroid/stress/atom`](https://docs.lammps.org/compute_heat_flux.html).
#    For systems with only two-body interactions, as in this notebook,
#    the `compute heat/flux` command with `compute stress/atom` command is sufficient.
#    This warning is important for molecular systems.

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(kappa_production - 7.05) > 0.1:
    raise ValueError(f"wrong thermal conductivity (production): {kappa_production:.3e}")
