# %% [markdown]
# # Shear viscosity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)
#
# This example shows how to calculate viscosity of argon
# from pressure tensor data obtained from {term}`LAMMPS` {term}`MD` simulations.
# The required theoretical background is explained the
# [](../properties/shear_viscosity.md) section.
# The same simulations are also used for the [bulk viscosity](lj_bulk_viscosity.py)
# and [thermal conductivity](lj_thermal_conductivity.py) examples in the following two notebooks.
# The goal of the argon examples is to derive the three transport properties
# with a relative error smaller than those found in the literature.

# All argon MD simulations use the
# [Lennard-Jones potential](https://en.wikipedia.org/wiki/Lennard-Jones_potential)
# with reduced Lennard-Jones units.
# For example, the reduced unit of viscosity is denoted as η\*,
# and the reduced unit of time as τ\*.
# The simulated system consists of 1372 argon atoms.
# The thermodynamic state $\rho=0.8442\,\mathrm{\rho}^*$ and $T=0.722\,\mathrm{T}^*$
# corresponds to a liquid phase near the triple point
# ($\rho=0.0845\,\mathrm{\rho}^*$ and $T=0.69\,\mathrm{T}^*$).
# This liquid state is known to exhibit slow relaxation times,
# which complicates the convergence of transport properties and
# makes it a popular test case for computational methods.
#
# The LAMMPS input files can be found in the directory `docs/data/lammps_lj3d`
# of STACIE's Git source repository.
# To obtain sufficient data for all three properties, we performed 100 independent runs,
# for which the [guesstimated relative error](../preparing_inputs/data_sufficiency.md)
# is tabulated below.
# The [Pade model](#section-pade-target) is used to fit the spectrum,
# with degrees $S_\text{num}=\{0, 2\}$ and $S_\text{den}=\{2\}$, corresponding to $P=3$ parameters.
#
# | Property             | $M$ | Guess rel. error |
# | -------------------- | --: | ---------------: |
# | Shear viscosity      | 500 | 0.5 %            |
# | Bulk viscosity       | 100 | 1.2 %            |
# | Thermal conductivity | 300 | 0.7 %            |
#
# The (initial) settings for the production runs were determined as follows.
# In general, the integration time step in MD simulations roughly
# corresponds to one tenth of a period of the fastest oscillations in the system.
# At shorter time scales than 10 steps, the dynamics is most likely irrelevant for transport properties.
# Hence, in our first simulations, all data was recorded with block averages of 10 steps.
# As mentioned in the section on the [block averages](../preparing_inputs/block_averages.md),
# at least $400 P$ blocks are recommended.
# The initial production runs therefore consisted of 12000 MD steps.
# Note that these values are only coarse estimates.
# As explained below, the production runs were extended twice to improve the statistics.
#
# Details of the MD simulations can be found the LAMMPS inputs
# `docs/data/lammps_lj3d/template-init.lammps` and `docs/data/lammps_lj3d/template-ext.lammps`
# in STACIE's Git repository.
# These input files are actually [Jinja2](https://jinja.palletsprojects.com/) templates
# that are rendered with different random seeds (and restart files) for each run.
# The initial production simulations start from an FCC crystal structure,
# which is first melted for 5000 steps
# at an elevated temperature of $T=1.5\,\mathrm{T}^*$ in the {term}`NVT` ensemble.
# The system is then equilibrated at the desired temperature
# of $T=0.722\,\mathrm{T}^*$ for 5000 additional steps.
# Starting from the equilibrated states, production runs were performed in the {term}`NVE` ensemble.
# The velocities are not rescaled after the NVT equilibration,
# to ensure that the set of NVE runs as a whole is representative of the NVT ensemble.
# During the production phase, trajectory data is collected with block averages over 10 steps.
#
# The LAMMPS input files contain commands to write output files
# that can be directly loaded using Python and NumPy without any additional converters or wrappers.
# The following output files from `docs/data/lammps_lj3d/sims/replica_????_part_??/` were used for the analysis:
#
# - `info.yaml`: simulation settings that may be useful for post-processing.
# - `nve_thermo.txt`: subsampled instantaneous temperature and related quantities
# - `nve_pressure_blav.txt`: block-averaged (off)diagonal pressure tensor components
# - `nve_heatflux_blav.txt`: block-averaged $x$, $y$, and $z$ components of
#   the heat flux vector, i.e. $J^\text{h}_x$, $J^\text{h}_y$, and $J^\text{h}_z$.
#   Heat fluxes are used in the thermal conductivity example, not in this notebook.
#
# :::{note}
# The results in this example were obtained using
# [LAMMPS version 4 Feb 2025](https://github.com/lammps/lammps/releases/tag/patch_4Feb2025).
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
from scipy.stats import chi2
from stacie import UnitConfig, compute_spectrum, estimate_acint, PadeModel
from stacie.plot import plot_fitted_spectrum, plot_extras
from utils import plot_instantaneous_percentiles, plot_cumulative_temperature_histogram

# %%
mpl.rc_file("matplotlibrc")
# %config InlineBackend.figure_formats = ["svg"]

# %%
# You normally do not need to change this path.
# It only needs to be overriden when building the documentation.
DATA_ROOT = Path(os.getenv("DATA_ROOT", "./")) / "lammps_lj3d/sims/"

# %% [markdown]
# ## Analysis of the Equilibration Runs
#
# To ensure that the production runs start from a well-equilibrated state,
# we first analyze the equilibration runs.
# The following code cell plots percentiles of the instantaneous temperature
# as a function of time over all independent runs.
# For reference, the theoretical percentiles of the NVT ensemble
# are shown as horizontal dotted lines.


# %%
def plot_equilibration(ntraj: int = 100):
    """Plot cumulative distributions of the instantaneous temperature."""
    # Load the configuration from the YAML file.
    with open(DATA_ROOT / "replica_0000_part_00/info.yaml") as fh:
        info = safe_load(fh)
    temp_d = info["temperature"]
    ndof = info["natom"] * 3 - 3

    # Load trajectory data.
    temps = []
    time = None
    for itraj in range(ntraj):
        equil_dir = DATA_ROOT / f"replica_{itraj:04d}_part_00/"
        data = np.loadtxt(equil_dir / "nvt_thermo.txt")
        temps.append(data[:, 1])
        if time is None:
            time = info["block_size"] * info["timestep"] * np.arange(len(data))
    temps = np.array(temps)

    # Select the last part (final temperature), discarding the melting phase.
    temps = temps[:, 550:]
    time = time[550:]

    # Plot the instantaneous and desired temperature.
    plt.close("tempequil")
    _, ax = plt.subplots(num="tempequil")
    percents = [95, 80, 50, 20, 5]
    plot_instantaneous_percentiles(
        ax,
        time,
        temps,
        percents,
        expected=[chi2.ppf(percent / 100, ndof) * temp_d / ndof for percent in percents],
    )
    ax.set_title(
        "Percentiles of the instantaneous temperature over the 100 equilibration runs"
    )
    ax.set_ylabel("Temperature")
    ax.set_xlabel("Time")


plot_equilibration()

# %% [markdown]
# The plot shows that the equilibration runs were successful:
# They reach the correct average temperature **and** also exhibit the expected fluctuations.
# Note that we used a Langevin thermestat for equilibration.
# This is a robust local thermostat that quickly brings
# all degrees of freedom to the desired temperature.
# In comparison, a global Nosé-Hoover-chain (NHC) thermostat would still show large
# oscillations in the temperature, even after 5000 steps.
# Taking the last state from an NHC run generally results in biased initial conditions for the NVE runs.
# (You can see the difference by modifying the LAMMPS input files,
# rerunning them and then rerunning this notebook.)

# %% [markdown]
# ## Analysis of the Initial Production Simulations
#
# The following code cell defines analysis functions:
#
# - `get_indep_paniso` transforms the pressure tensor components
#   into five independent anisotropic contributions,
#   as explained in the [](../properties/shear_viscosity.md) theory section.
# - `estimate_viscosity` calculates the viscosity and plots the results.
#   It also prints recommendations for data reduction (block averaging) and simulation time,
#   as explained in the following two sections of the documentation:
#
#     - [](../properties/autocorrelation_time.md)
#     - [](../preparing_inputs/block_averages.md)
#
#     These will be used to determine whether our initial simulation settings are appropriate.


# %%
def get_indep_paniso(pcomps):
    return np.array(
        [
            (pcomps[0] - 0.5 * pcomps[1] - 0.5 * pcomps[2]) / np.sqrt(3),
            0.5 * pcomps[1] - 0.5 * pcomps[2],
            pcomps[3],
            pcomps[4],
            pcomps[5],
        ]
    )


def estimate_viscosity(name, pcomps, av_temperature, volume, timestep, verbose=True):
    # Create the spectrum of the pressure fluctuations.
    # Note that the Boltzmann constant is 1 in reduced LJ units.
    uc = UnitConfig(
        acint_fmt=".3f",
        acint_symbol="η",
        acint_unit_str="η*",
        freq_unit_str="1/τ*",
        time_fmt=".3f",
        time_unit_str="τ*",
    )
    spectrum = compute_spectrum(
        pcomps,
        prefactors=volume / av_temperature,
        timestep=timestep,
    )

    # Estimate the viscosity from the spectrum.
    result = estimate_acint(spectrum, PadeModel([0, 2], [2]), verbose=verbose, uc=uc)

    if verbose:
        # Plot some basic analysis figures.
        plt.close(f"{name}_spectrum")
        _, ax = plt.subplots(num=f"{name}_spectrum")
        plot_fitted_spectrum(ax, uc, result)
        plt.close(f"{name}_extras")
        _, axs = plt.subplots(2, 2, num=f"{name}_extras")
        plot_extras(axs, uc, result)

    # Return the viscosity
    return result


# %% [markdown]
# The next cell performs the analysis of the initial simulations.
# It prints the recommended block size and the simulation time for the production runs,
# and then generates two figures:
#
# - The spectrum of the off-diagonal pressure fluctuations, and the model fitted to the spectrum.
# - Additional intermediate results.


# %%
def demo_production(npart: int, ntraj: int = 100, select: int | None = None):
    """
    Perform the analysis of the production runs.

    Parameters
    ----------
    npart
        Number of parts in the production runs.
        For the initial production runs, this is 1.
    ntraj
        Number of trajectories in the production runs.
    select
        If `None`, all anisotropic contributions are selected.
        If not `None`, only select the given anisotropic contribution
        for the viscosity estimate. Must be one of `0`, `1`, `2`, `3`, `4`, `None`.

    Returns
    -------
    result
        The result from STACIE's `estimate_acint` function,
    """
    # Load the configuration from the YAML file.
    with open(DATA_ROOT / "replica_0000_part_00/info.yaml") as fh:
        info = safe_load(fh)

    # Load trajectory data.
    thermos = []
    pcomps_full = []
    for itraj in range(ntraj):
        thermos.append([])
        pcomps_full.append([])
        for ipart in range(npart):
            prod_dir = DATA_ROOT / f"replica_{itraj:04d}_part_{ipart:02d}/"
            thermos[-1].append(np.loadtxt(prod_dir / "nve_thermo.txt"))
            pcomps_full[-1].append(np.loadtxt(prod_dir / "nve_pressure_blav.txt"))
    thermos = [np.concatenate(parts).T for parts in thermos]
    pcomps_full = [np.concatenate(parts).T for parts in pcomps_full]
    av_temperature = np.mean([thermo[1] for thermo in thermos])

    # Compute the viscosity.
    pcomps_aniso = np.concatenate([get_indep_paniso(p[1:]) for p in pcomps_full])
    if select is not None:
        if select < 0 or select > 4:
            raise ValueError(f"Invalid selection {select}, must be in [0, 4]")
        pcomps_aniso = pcomps_aniso[select::5]
    return estimate_viscosity(
        f"part{npart}",
        pcomps_aniso,
        av_temperature,
        info["volume"],
        info["timestep"] * info["block_size"],
        verbose=select is None,
    )


eta_production_init = demo_production(1).acint


# %% [markdown]
# Several things can be observed in the analysis of the initial production runs:
#
# - The recommendations based on the exponential correlation time were met by the initial simulation settings.
#
#     - The recommended simulation time is 27 τ\*, which about 9000 steps.
#       The initial production runs (12000 steps) were therefore sufficient.
#
#     - The recommended block size is 0.057 τ\*, which corresponds to about 45 steps.
#       The block size used in the initial production runs (10 steps) was sufficiently small.
#
# - The relative error of the viscosity estimate is 2.2%,
#   which is larger than the guesstimated value 0.5%.
#   This is fine and somewhat expected, since this guess is known to be crude.
#
# - The Pade model used to fit the spectrum was a fair choice, but for higher frequencies,
#   the sampling PSD clearly decays faster than the fitted model.
#   For the case of viscosity, there is (to the best of our knowledge) no solid theoretical argument
#   to support the exponential decay of the ACF of the pressure tensor.
#   It just seems to be a reasonable choice for this case.
#
# - The effective number of points fitted to the spectrum is 19.4,
#   which is low for a 3 parameter model.
#   For high-quality production simulations, it would be good to triple the simulation length,
#   as to multiply the resolution of the frequency grid by 3.
#   This should lead to approximately 60 effective points.
#
# As can be seen in the comparison to literature results below,
# the results for the initial production runs were already quite good.
# However, for the sake of demonstration,
# the production runs were extended by an additional 24000 steps each,
# to triple the simulation time.
# This revealed that the effective number of points fitted to the spectrum
# did not increase accordingly (only up to 43 points).
# Hence we decided to extend the production runs by another 64000 steps,
# which resulted in a total simulation time of 300 τ\* per run.
#
# The difficulty of increasing the effective number of fitted points can be understood as follows.
# The Pade model is not capable of fitting the spectrum to higher frequencies.
# By including more data points, the limitations of the approximating model also become clearer,
# and the cutoff criterion will detect some underfitting (and thus risk for bias) at lower cutoffs.

# %% [markdown]
# ## Analysis of the Production Simulations
#
# Here we just repeat the analysis, but now with extended production runs.

# %%
eta_production_ext = demo_production(3).acint

# %% [markdown]
# Some remarks about the final results:
#
# - The effective number of points has increased to 94.2,
#   which is a fine number of data points for a model with $P=3$ parameters.
#
# - For the lowest frequency cutoffs the parameters show overfitting artifacts.
#   Visually, one can also see that the first few data points in the spectrum are
#   slightly lower compared to points at higher frequencies.
#   This is (most likely) a statistical artifact.
#   To reduce the risk that such artifacts affect the final result,
#   one can increase the parameter `neff_min`, which is $5\times P$ by default.
#   Hence, at the lowest frequency cutoff, the effective number of points is 15.


# %% [markdown]
# ## Comparison to Literature Results
#
# Comprehensive literature surveys on computational estimates of the shear viscosity of a Lennard-Jones fluid
# can be found in {cite:p}`meier_2004_transport_I` and {cite:p}`viscardi_2007_transport1`.
# These papers also present new results, which are included in the table below.
# Since the simulation settings ($r_\text{cut}^{*}=2.5$, $N=1372$, $T^*=0.722$ and $\rho^{*}=0.8442$)
# are identical to those used in this notebook, the reported values should be directly comparable.
#
# | Method                     | Simulation time [τ\*] | Shear viscosity [η\*] | Reference |
# |----------------------------| --------------------: | --------------------: |-----------|
# | EMD NVE (STACIE)           | 3600                  | 3.236 ± 0.072         | (here) initial |
# | EMD NVE (STACIE)           | 10800                 | 3.232 ± 0.049         | (here) extension 1 |
# | EMD NVE (STACIE)           | 30000                 | 3.257 ± 0.030         | (here) extension 2 |
# | EMD NVE (Helfand-Einstein) | 75000                 | 3.277 ± 0.098         | {cite:p}`meier_2004_transport_I` |
# | EMD NVE (Helfand-moment)   | 600000                | 3.268 ± 0.055         | {cite:p}`viscardi_2007_transport1` |
#
# This comparison confirms that STACIE can reproduce a well-known viscosity result,
# in line with literature results.
# To achieve a state-of-the-art statistical uncertainty, it requires far less simulation time.
# Even our longest production runs are still less than half as long as in the cited papers,
# and we achieve a much smaller standard error.
#
# To be fair, the simulation time only accounts for production runs.
# Our setup also includes a significant amount of equilibration runs (3000 τ* in total)
# to ensure that different production runs are uncorrelated.
# Even when these additional runs are included, the overall simulation time
# remains significantly lower than in the cited papers.
#

# %% [markdown]
# ## Analysis of the Temperature Distribution of the Production Runs
#
# To further establish the validity of the claim that our NVE runs together represent the
# NVT ensemble, the following cell analyzes the distribution of the instantaneous temperature.
# For each individual NVE run and for the combined NVE runs, cumulative distributions are plotted.
# The function also plots the expected cumulative distribution of the NVT ensemble.


# %%
def validate_temperature():
    """Plot cumulative distributions of the instantaneous temperature."""
    # Load the configuration from the YAML file.
    with open(DATA_ROOT / "replica_0000_part_00/info.yaml") as fh:
        info = safe_load(fh)
    temp_d = info["temperature"]
    ndof = info["natom"] * 3 - 3

    # Load trajectory data.
    temps = []
    for itraj in range(100):
        temps.append([])
        for ipart in range(3):
            prod_dir = DATA_ROOT / f"replica_{itraj:04d}_part_{ipart:02d}/"
            temps[-1].append(np.loadtxt(prod_dir / "nve_thermo.txt")[:, 1])
    temps = [np.concatenate(temp).T for temp in temps]

    # Plot the instantaneous and desired temperature distribution.
    plt.close("tempprod")
    _, ax = plt.subplots(num="tempprod")
    plot_cumulative_temperature_histogram(ax, temps, temp_d, ndof, "τ*")


validate_temperature()

# %% [markdown]
#
# This plot offers detailed insight into NVE versus NVT temperature distributions:
#
# - In the NVE ensemble, the temperature distribution is relatively narrow.
#   Hence, using a single NVE run would not be representative of the temperature variance of the NVT ensemble.
# - Some of the individual NVE runs have significantly lower ore higher temperatures than the average.
#   If the transport property of interest has a nonlinear dependence on the temperature,
#   this effect will lead to a shift in the estimated transport property,
#   compared to using a single NVE run.
#
# In the limit of macroscopic system sizes ($N \rightarrow \infty$),
# the NVE ensemble converges to the NVT ensemble.
# However, in simulations, one operates at finite system sizes,
# well below the thermodynamic limit.

# %% [markdown]
# ## Validation of the Independence of the Anistropic Contributions
#
# Here we validate numerically that the
# [five independent anisotropic contributions](../properties/shear_viscosity.md)
# to the pressure tensor are indeed statistically independent.
# The covariance matrix of the anisotropic contributions is computed and the off-diagonal elements
# are plotted.


# %%
def validate_independence(ntraj: int = 100):
    """Validate the independence of the anisotropic contributions."""
    # Load trajectory data.
    pcomps_full = []
    for itraj in range(ntraj):
        pcomps_full.append([])
        for ipart in range(3):
            prod_dir = DATA_ROOT / f"replica_{itraj:04d}_part_{ipart:02d}/"
            pcomps_full[-1].append(np.loadtxt(prod_dir / "nve_pressure_blav.txt")[:, 1:])
    pcomps_aniso = [get_indep_paniso(np.concatenate(parts).T) for parts in pcomps_full]

    # Compute the average of the covariance matrix over all NVE trajectories.
    cov = np.mean([np.cov(p, ddof=0) for p in pcomps_aniso], axis=0)
    scale = abs(cov).max() * 1.05

    # Plot the covariance matrix.
    plt.close("covariance")
    _, ax = plt.subplots(num="covariance")
    im = ax.imshow(
        cov, cmap="coolwarm", vmin=-scale, vmax=scale, extent=[0.5, 5.5, 0.5, 5.5]
    )
    ax.set_title("Covariance of Anisotropic Pressure Contributions")
    ax.set_xlabel("Anisotropic Contribution $P'_i$")
    ax.set_ylabel("Anisotropic Contribution $P'_j$")
    plt.colorbar(im, ax=ax)


validate_independence()

# %% [markdown]
# The plot confirms that there is (at least visually)
# no sign of any statistical correlation between the anisotropic contributions.
# Note that one may perform more rigorous statistical tests
# to validate the independence of the five contributions.
# Here, we keep it simple for the sake of an intuitive demonstration.

# %% [markdown]
# ## Validation of the consistency of the Anisotropic Contributions
#
# The following code cell shows that the five independent anisotropic contributions
# result in the same shear viscosity estimate, within the predicted uncertainties.


# %%
def validate_consistency():
    for i in range(5):
        result = demo_production(3, select=i)
        eta = result.acint
        eta_std = result.acint_std
        print(f"Anisotropic contribution {i + 1}: η = {eta:.3f} ± {eta_std:.3f} η*")


validate_consistency()

# %% [markdown]
# Note that one may perform more rigorous statistical tests to validate the consistency of the results.
# Here, we keep it simple for the sake of an intuitive demonstration.

# %%  [markdown]
# ## Regression Tests
#
# If you are experimenting with this notebook, you can ignore any exceptions below.
# The tests are only meant to pass for the notebook in its original form.

# %%
if abs(eta_production_init - 3.236) > 0.1:
    raise ValueError(f"wrong viscosity (production): {eta_production_init:.3e}")
if abs(eta_production_ext - 3.257) > 0.1:
    raise ValueError(f"wrong viscosity (production): {eta_production_ext:.3e}")
