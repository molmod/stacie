# LAMMPS 3D Lennard-Jones Simulations

This dataset contains LAMMPS simulations of a 3D Lennard-Jones system
used to demonstrate the calculation of shear and bulk viscosity, and thermal conductivity
in the STACIE documentation.

**Tip:** The [Lammps Syntax Highlighting](https://marketplace.visualstudio.com/items?itemName=ThFriedrich.lammps)
greatly facilitates understanding and authoring LAMMPS input files.

## Software Requirements

You must have LAMMPS installed to run the simulations.
All other requirements are installed in a Python virtual environment with:

```bash
pip install stepup
```

## File and Directory Summary

- `plan.py`: The workflow plan for StepUp.
- `runlammps.py`: A Python wrapper for LAMMPS to facilitate the integration with StepUp.
- `sims/`: Directory where all output files are saved.
- `template-ext.lammps`: Template for the extension of the NVE production simulation.
- `template-init.lammps`: Template for the equilibration and the initial NVE production simulation.

## File Content Details

The LAMMPS input files contain instructions to write all relevant simulation data to text files
that can be loaded easily with `numpy.loadtxt` and `yaml.safe_load`.
No specialized LAMMPS Python libraries are required to read the simulation outputs.
Note that Numpy and PyYAML are not needed to run the simulations,
but only to read the outputs.

## Data Generation

All LAMMPS simulations can be executed efficiently as follows:

```bash
stepup boot -n 1.0
```

This will create input files from the templates `template-init.lammps` and `template-ext.lammps`
and run the LAMMPS simulations in parallel multiple times
with the correct arguments and in the right order.
All simulation outputs will be saved in the `sims/` directory.

Because StepUp already executes the workflow in parallel, serial LAMMPS calculations are used.
The inputs were tested with the LAMMPS release of 4 Feb 2025.

It is recommended to run the workflow on a compute cluster.
