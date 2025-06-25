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

- `minimal.py`: Minimal example of how to use STACIE, with detailed description of outputs.
- `error_mean.py`: Uncertainty of the mean of time-correlated data
- `applicability.py`: Applicability of the Lorentz model
- `surface_diffusion.py`: Diffusion of an argon atom on a surface

Examples that analyze existing trajectory data:

- `lj_shear_viscosity.py`: Shear viscosity of a Lennard-Jones fluid
- `lj_bulk_viscosity.py`: Bulk viscosity of a Lennard-Jones fluid
- `lj_thermal_conductivity.py`: Thermal conductivity of a Lennard-Jones fluid
- `molten_salt.py`: Ionic electrical conductivity of a molten salt system

This second set of notebooks use MD data from the following sources:

- `lammps_lj3d`: LAMMPS simulations of Lennard-Jones 3D systems
- `openmm_salt`: OpenMM simulations of molten salt systems
