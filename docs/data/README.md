# Input data for STACIE's worked examples

See `README.typ` for a description of how to use this data.

## Regeneration of the MD outputs from their input files

To rerun the MD simulations, you need files that are not included in the Zenodo archive,
such as the LAMMPS and OpenMM input files, and the force fields.
These are available in the STACIE Git repository: <https://github.com/molmod/stacie>

Each directory under `docs/data` contains a `plan.py` script that can be executed with `stepup boot`
to run all simulations in parallel.
See the `README.md` file in each directory for more details.

In addition, the top-level directory contains a `plan.py` script to archive the examples,
Jupyter notebooks, and data files into a single ZIP file for upload to Zenodo.
(This is only relevant for STACIE developers.)
