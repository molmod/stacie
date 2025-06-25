# %% [markdown]
# # Bulk Viscosity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)
#
# This example demonstrates how to compute the bulk viscosity
# of a Lennard-Jones liquid near its triple point using LAMMPS.
# It uses the same production runs and conventions
# as in the [Shear viscosity example](lj_shear_viscosity.py).
# The required theoretical background is explained the section
# [](../properties/bulk_viscosity.md).
# In essence, it is computed in the same way as the shear viscosity,
# except that the isotropic pressure fluctuations are used as input.
#
# :::{note}
# The results in this example were obtained using
# [LAMMPS 29 Aug 2024 Update 3](https://github.com/lammps/lammps/releases/tag/stable_29Aug2024_update3).
# Minor differences may arise when using a different version of LAMMPS,
# or even the same version compiled with a different compiler.
# :::

# %% [markdown]
# ## Library Imports and Matplotlib Configuration

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
# The following code cells define analysis functions used below.
#
# - `get_piso`: Computes the isotropic pressure from the diagonal components
#   of the time-dependent pressure tensor ($P_{xx}$, $P_{yy}$, and $P_{zz}$),
#   as explained in the [bulk viscosity](../properties/bulk_viscosity.md) theory section.
# - `estimate_bulk_viscosity`: Computes the bulk viscosity, visualizes the results,
#   and provides recommendations for data reduction (block averaging) and simulation time,
#   as explained in the following two sections of the documentation:
#     - [](../properties/autocorrelation_time.md)
#     - [](../preparing_inputs/block_averages.md)


# %%
def estimate_bulk_viscosity(name, p_iso, av_temperature, volume, timestep):
    # Compute spectrum of the isotropic pressure fluctuations.
    # Note that the Boltzmann constant is 1 in reduced LJ units.
    uc = UnitConfig(
        acint_fmt=".3f",
        acint_symbol="η_b",
        acint_unit_str="η*",
        freq_unit_str="1/τ*",
        time_fmt=".3f",
        time_unit_str="τ*",
    )
    spectrum = compute_spectrum(
        p_iso,
        prefactors=volume / av_temperature,
        timestep=timestep,
        include_zero_freq=False,
    )

    # Estimate the bulk viscosity from the spectrum.
    result = estimate_acint(spectrum, LorentzModel(), verbose=True, uc=uc)

    # Plot some basic analysis figures.
    plt.close(f"{name}_spectrum")
    _, ax = plt.subplots(num=f"{name}_spectrum")
    plot_fitted_spectrum(ax, uc, result)
    plt.close(f"{name}_extras")
    _, axs = plt.subplots(2, 2, num=f"{name}_extras")
    plot_extras(axs, uc, result)

    # Return the bulk viscosity
    return result.acint


# %% [markdown]
# :::{note}
# When computing bulk viscosity, the `include_zero_freq` argument in
# the `compute_spectrum` function must be set to `False`,
# as the average pressure is nonzero.
# This ensures the DC component is excluded from the spectrum.
# See the [bulk viscosity](../properties/bulk_viscosity.md) theory section for more details.
# :::


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
    eta_bulk
        The estimated bulk viscosity.
    """
    # Load the configuration from the YAML file.
    with open(DATA_ROOT / "replica_0000_part_00/info.yaml") as fh:
        info = safe_load(fh)

    # Load trajectory data.
    thermos = []
    p_isos = []
    for itraj in range(ntraj):
        thermos.append([])
        p_isos.append([])
        for ipart in range(npart):
            prod_dir = DATA_ROOT / f"replica_{itraj:04d}_part_{ipart:02d}/"
            thermos[-1].append(np.loadtxt(prod_dir / "nve_thermo.txt"))
            # The average over columns 2, 3 and 4 of parts corresponds
            # to the time-dependent isotropic pressure.
            p_comps = np.loadtxt(prod_dir / "nve_pressure_blav.txt")
            p_isos[-1].append(p_comps[:, 1:4].mean(axis=1))
    thermos = [np.concatenate(parts).T for parts in thermos]
    p_iso = [np.concatenate(parts) for parts in p_isos]

    # Compute the average temperature
    av_temperature = np.mean([thermo[1] for thermo in thermos])

    # Compute the bulk viscosity
    return estimate_bulk_viscosity(
        f"part{npart}",
        p_iso,
        av_temperature,
        info["volume"],
        info["timestep"] * info["block_size"],
    )


eta_bulk_production = demo_production(3)

# %% [markdown]
# The cutoff criterion Z-score is relatively high, around 2.
# This suggests that the fits on the two halves deviate more from each other
# than what would be expected from the {term}`uncertainty` of the spectrum.
# There are multiple potential explanations for this observation:
#
# - One potential explanation is that the isotropic pressure fluctuations
#   are not perfectly Gaussian.
#   This is expected for a Lennard-Jones fluid,
#   as expansion of the system will result in slightly lower restoring forces than compression.
#   Such a slightly non-Gaussian distribution of the pressure fluctuations
#   can result in a distribution of spectral data
#   that deviates from the Gamma distribution employed by STACIE.
#
# - Another potential cause is that there is not yet sufficient data to fix the cutoff frequency.
#   This can be addressed by generating more trajectory data,
#   which will make it easier to determine the suitable range of cutoff frequencies.
#   We have not further expanded the production runs in this example,
#   to keep the computational cost low.
#   Furthermore, as shown in the comparison below,
#   we already obtained a good agreement with the literature results
#   and relatively small uncertainties.

# %% [markdown]
# ## Comparison to Literature Results
#
# Computational estimates of the bulk viscosity of a Lennard-Jones fluid
# can be found in {cite:p}`meier_2004_transport_III`.
# Since the simulation settings ($r_\text{cut}^{*}=2.5$, $N=1372$, $T^*=0.722$ and $\rho^{*}=0.8442$)
# are identical to those used in this notebook, the reported values should be directly comparable.
#
# | Method                     | Simulation time [τ\*] | Bulk viscosity [η$_b$\*] | Reference |
# | -------------------------- | --------------------: | -----------------------: | --------- |
# | EMD NVE (STACIE)           | 10800                 | 1.158 ± 0.030            | (here) extension 1 |
# | EMD NVE (STACIE)           | 30000                 | 1.191 ± 0.021            | (here) extension 2 |
# | EMD NVE (Helfand-Einstein) | 300000                | 1.186 ± 0.084            | {cite:p}`meier_2004_transport_III` |
#
# This comparison demonstrates that STACIE accurately reproduces bulk viscosity results
# while achieving lower statistical uncertainty with significantly less data than existing methods.
#
# Note that the results for only the initial NVE production run are not included because
# the sanity checks indicated that the data was not sufficient.

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(eta_bulk_production - 1.195) > 0.1:
    raise ValueError(f"wrong viscosity (production): {eta_bulk_production:.3e}")
