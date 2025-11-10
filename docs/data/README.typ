#show link: set text(blue)
#set page("a4", margin: 2cm)

#align(center)[
  #text(size: 24pt)[
    *Example Trajectory Data and Jupyter Notebooks Showing How to Compute Various Properties with STACIE*
  ]

  Gözdenur Toraman#super[†] and Toon Verstraelen#super[✶¶]

  † Soete Laboratory, Ghent University, Technologiepark-Zwijnaarde 46, 9052 Ghent, Belgium\
  ¶ Center for Molecular Modeling (CMM), Ghent University, Technologiepark-Zwijnaarde
  46, B-9052, Ghent, Belgium

  ✶E-mail: #link("mailto:toon.verstraelen@ugent.be", "toon.verstraelen@ugent.be")
]

== Usage

To run the example notebooks, you need to:

1. Install STACIE and Jupyter Lab

    ```bash
    pip install stacie jupyterlab
    ```

2. Download and unpack the archive with notebooks and trajectory data.

    ```bash
    unzip examples.zip
    ```

3. Finally, you should be able to start Jupyter Lab and run the notebooks.

    ```bash
    jupyter lab
    ```

== Overview of included files

Some Jupyter notebooks generate data and then analyze it, while others
directly analyze existing data.

Examples that do not need existing trajectory data:

- `minimal.ipynb`: Minimal example of how to use STACIE, with detailed description of outputs.
- `error_mean.ipynb`: Uncertainty of the mean of time-correlated data
- `applicability.ipynb`: Applicability of the Lorentz model
- `surface_diffusion.ipynb`: Diffusion of an argon atom on a surface

Examples that analyze existing molecular dynamics trajectory data:

- `lj_shear_viscosity.ipynb`: Shear viscosity of a Lennard-Jones fluid
- `lj_bulk_viscosity.ipynb`: Bulk viscosity of a Lennard-Jones fluid
- `lj_thermal_conductivity.ipynb`: Thermal conductivity of a Lennard-Jones fluid
- `molten_salt.ipynb`: Ionic electrical conductivity of a molten salt system

This second set of notebooks use MD data from the following sources:

- `lammps_lj3d`: LAMMPS simulations of Lennard-Jones 3D systems
- `openmm_salt`: OpenMM simulations of molten salt systems

Examples analyzing data from external sources:

- `cloud-cover.ipynb`: Correlation time analysis of cloud cover data from Open-Meteo

== Revision history

=== v1.1.0 (2025-11-10)

Include cloud cover example data and notebook.

=== v1.0.2 (2025-10-18)

Switch to choice of license `CC-BY-SA-4.0 OR LGPL-3.0-or-later`.

=== v1.0.1 (2025-10-03)

Bugfix release: add missing `nvt_thermo.txt` files to the `lammps_lj3d` data.

=== v1.0.0 (2025-06-26)

Initial release of the STACIE example data and notebooks.
