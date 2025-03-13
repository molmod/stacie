#!/usr/bin/env python

# %% [markdown]
# # Bulk viscosity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)
#
# This example demonstrates how to compute the bulk viscosity
# of a Lennard-Jones liquid near its triple point using LAMMPS.
# It is based on the previous notebook [Shear viscosity of the Lennard-Jones liquid](lj_viscosity.py),
# but skips the exploration phase to avoid repetition, focusing only on production simulations.
# In this case, isotropic pressure is used to determine the bulk viscosity.

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from path import Path
from yaml import safe_load
from stacie import UnitConfig, compute_spectrum, estimate_acint, summarize_results
from stacie.plot import plot_fitted_spectrum, plot_criterion
from IPython.display import display, HTML

display(HTML("<style> .note { text-align: justify; } </style>"))

# %%
mpl.rc_file("matplotlibrc")

# %% [markdown]
# The following code cells define key analysis functions used in
# production simulations.
#
# - `get_piso`: Computes the isotropic pressure from the diagonal components
#   of the time-dependent pressure tensor ($P_{xx}$, $P_{yy}$, and $P_{zz}$),
#   as explained in the [bulk viscosity](../theory/properties/bulk_viscosity.md) theory section.
# - `estimate_bulk_viscosity`: Computes the bulk viscosity, visualizes the results,
#   and provides recommendations for data reduction (block averaging) and simulation time,
#   as explained in the following two sections of the documentation:
#     - [Autocorrelation Time](../theory/properties/autocorrelation_time.md)
#     - [Block Averages](../theory/advanced_topics/block_averages.md)


# %%
def get_piso(pcomps):
    return np.array([(pcomps[0] + pcomps[1] + pcomps[2]) / 3])


def estimate_bulk_viscosity(name, pcomps, av_temperature, volume, timestep):
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
        pcomps,
        prefactor=0.5 * volume / av_temperature,
        timestep=timestep,
        include_zero_freq=False,
    )

    # Estimate the bulk viscosity from the spectrum.
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

    # Return the bulk viscosity
    return result.acint


# %% [markdown]
# ::: {note}
# When computing bulk viscosity, the `include_zero_freq` argument in
# the `compute_spectrum` function must be set to `False`,
# as the average pressure is nonzero.
# This ensures the DC component is excluded from the spectrum.
# See the [bulk viscosity](../theory/properties/bulk_viscosity.md) theory section for more details.


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
    # - `part_01` corresponds to the first extension of the production run
    #   (4000 extra steps).
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

    # Compute the average temperature
    av_temperature = np.mean([thermo[1] for thermo in thermos])

    # Compute the bulk viscosity
    pcomps = np.concatenate([get_piso(pressure[1:]) for pressure in pressures])
    return estimate_bulk_viscosity(
        name,
        pcomps,
        av_temperature,
        info["volume"],
        info["timestep"] * info["block_size"],
    )


eta_bulk_production = demo_production()

# %% [markdown]
# ## Comparison to Literature Results
#
# Computational estimates of the bulk viscosity of a Lennard-Jones fluid
# can be found in {cite:p}`meier_2004_transport_III`.
# Since the simulation settings ($r_\text{cut}^{*}=2.5$, $N=1372$, $T^*=0.722$ and $\rho^{*}=0.8442$)
# are identical to those used in this notebook, the reported values should be directly comparable.
#
# | Method                     | Simulation time [τ\*] | Bulk viscosity [η$_b$\*] | Reference |
# |----------------------------|-----------------------|-----------------|-----------|
# | EMD NVE (Helfand-Einstein) | 300000                | 1.186 ± 0.084   | {cite:p}`meier_2004_transport_III` |
# | EMD NVE (Stacie)           | 2400                  | 1.131 ± 0.058   | This notebook |
#
# This comparison demonstrates that Stacie accurately reproduces bulk viscosity results
# while achieving lower statistical uncertainty with significantly less data than existing methods.
#
# ::: {note}
#
#     The results in this study were obtained using
#     [LAMMPS version 19 Nov 2024](https://github.com/lammps/lammps/releases/tag/patch_19Nov2024).
#     Note that minor differences may arise when using a different version of LAMMPS,
#     or even the same version compiled with a different compiler.

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(eta_bulk_production - 1.15) > 0.1:
    raise ValueError(f"wrong viscosity (production): {eta_bulk_production:.3e}")
