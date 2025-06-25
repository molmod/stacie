# %% [markdown]
# # Thermal Conductivity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)

# This example shows how to derive the thermal conductivity
# using heat flux data from a LAMMPS simulation.
# It uses the same production runs and conventions
# as in the [Shear viscosity example](lj_shear_viscosity.py).
# The required theoretical background is explained the section
# [](../properties/thermal_conductivity.md).
#
# :::{warning}
# A Lennard-Jones system only exhibits pairwise interactions,
# for which the LAMMPS command `compute/heat flux` produces valid results.
# For systems with three- or higher-body interactions, one cannot simply use the same command.
# Consult the theory section on [thermal conductivity](../properties/thermal_conductivity.md)
# for more background.
# :::
#
# :::{note}
# The results in this example were obtained using
# [LAMMPS 29 Aug 2024 Update 3](https://github.com/lammps/lammps/releases/tag/stable_29Aug2024_update3).
# Minor differences may arise when using a different version of LAMMPS,
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
from yaml import safe_load
from stacie import (
    UnitConfig,
    compute_spectrum,
    estimate_acint,
    LorentzModel,
    plot_fitted_spectrum,
    plot_extras,
)

# %%
mpl.rc_file("matplotlibrc")
# %config InlineBackend.figure_formats = ["svg"]

# %%
# You normally do not need to change this path.
# It only needs to be overridden when building the documentation.
DATA_ROOT = Path(os.getenv("DATA_ROOT", "./")) / "lammps_lj3d/sims/"

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
    result = estimate_acint(spectrum, LorentzModel(), verbose=True, uc=uc)

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
def demo_production(npart: int = 3, ntraj: int = 100):
    """
    Perform the analysis of the production runs.

    Parameters
    ----------
    npart
        Number of parts in the production runs.
        For the initial production runs, this is 1.
    ntraj
        Number of trajectories in the production runs.

    Returns
    -------
    kappa
        The estimated thermal conductivity.
    """
    # Load the configuration from the YAML file.
    with open(DATA_ROOT / "replica_0000_part_00/info.yaml") as fh:
        info = safe_load(fh)

    # Load trajectory data, without hardcoding the number of runs and parts.
    thermos = []
    heatfluxes = []
    for itraj in range(ntraj):
        thermos.append([])
        heatfluxes.append([])
        for ipart in range(npart):
            prod_dir = DATA_ROOT / f"replica_{itraj:04d}_part_{ipart:02d}/"
            thermos[-1].append(np.loadtxt(prod_dir / "nve_thermo.txt"))
            heatfluxes[-1].append(np.loadtxt(prod_dir / "nve_heatflux_blav.txt"))
    thermos = [np.concatenate(parts).T for parts in thermos]
    heatfluxes = [np.concatenate(parts).T for parts in heatfluxes]

    # Compute the average temperature.
    av_temperature = np.mean([thermo[1].mean() for thermo in thermos])

    # Compute the thermal conductivity.
    # Note that the last three columns are not used in the analysis.
    # According to the LAMMPS documentation, the last three columns
    # only contain the convective contribution to the heat flux.
    # See https://docs.lammps.org/compute_heat_flux.html
    jcomps = np.concatenate([heatflux[1:4] for heatflux in heatfluxes])
    return estimate_thermal_conductivity(
        f"part{npart}",
        jcomps,
        av_temperature,
        info["volume"],
        info["timestep"] * info["block_size"],
    )


kappa_production = demo_production(3)

# %% [markdown]
# The exponential correlation time of the heat flux tensor fluctuations is five times shorter
# than that of the pressure tensor fluctuations.
# This means that the thermal conductivity is a bit easier to compute than the viscosity.
# Note that the selected block size is still compatible with this shorter time scale.
#
# Similarly to the bulk viscosity, the Z-scores are clearly positive.
# This could be for the same reasons as in the bulk viscosity example.
# In addition, the block size of 0.03 τ* is slightly larger than the recommended 0.028 τ*,
# meaning that the spectrum might be perturbed by (very) small aliasing effects
# that could distort the fit.

# %% [markdown]
# ## Comparison to Literature Results
#
# A detailed literature survey of computational estimates of the thermal conductivity
# of a Lennard-Jones fluid can be found in {cite:p}`viscardi_2007_transport2`.
# Viscardi also computes new estimates, one of which is included in the table below.
# This value can be directly comparable to the current notebook,
# because the settings are identical
# ($r_\text{cut}^{*}=2.5$, $N=1372$, $T^*=0.722$ and $\rho^{*}=0.8442$).
#
# | Method                     | Simulation time  [τ\*] | Thermal conductivity [κ\*] | Reference |
# |----------------------------|------------------------|----------------------------|-----------|
# | EMD NVE (STACIE)           | 3600                   | 6.837 ± 0.081              | (here) initial |
# | EMD NVE (STACIE)           | 10800                  | 6.968 ± 0.046              | (here) extension 1 |
# | EMD NVE (STACIE)           | 30000                  | 6.936 ± 0.029              | (here) extension 2 |
# | EMD NVE (Helfand-moment)   | 600000                 | 6.946 ± 0.12               | {cite:p}`viscardi_2007_transport2` |
#
# This small comparison confirms that STACIE can reproduce a well-known thermal conductivity result,
# with small error bars, while using much less trajectory data than existing methods.

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(kappa_production - 6.953) > 0.2:
    raise ValueError(f"wrong thermal conductivity (production): {kappa_production:.3e}")
