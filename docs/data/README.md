# Input data for STACIE's worked examples

This directory contains the input files needed to run simulations that generate data
for the worked examples in the STACIE documentation.
There are currently two sets of simulations:

- `lammps_lj3d`: LAMMPS simulations of Lennard-Jones 3D systems
- `openmm_salt`: OpenMM simulations of molten salt systems

Each directory contains a `plan.py` script that can be executed with `stepup boot`
to run all simulations in parallel.
See the README.md file in each directory for more details.

In addition, the top-level directory contains a `plan.py` script to archive the examples,
Jupyter notebooks, and data files into a single ZIP file for upload to Zenodo.
(This is only relevant for STACIE developers.)
