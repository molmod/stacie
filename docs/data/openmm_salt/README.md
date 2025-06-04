# OpenMM Molten Salt Simulations

This dataset contains the OpenMM simulations of molten salts
used to demonstrate the conductivity calculation in the STACIE documentation.

## Software Requirements

To repeat the computations, you need OpenMM, MDTraj, NGLView, NumPy and Pandas,
which are included in the documentation dependencies of STACIE.
To run the workflow with StepUp, you also need StepUp and StepUp RepRep.
You can install all required dependencies as follows,
assuming you have set up a Python virtual environment:

```bash
pip install stacie[docs,tests] stepup-reprep
```

## File and Directory Summary

The relevant files are:

- `bmhtf.py`: Python script that defines force field in OpenMM.
- `extension.ipynb`: Notebook that extends the initial simulation with additional NVE steps.
- `extract.py`: Python script that extracts the simulation data from the OpenMM trajectory files.
  It reads PDB and DCD files and saves the data in NPZ format.
- `initial.ipynb`: Notebook to equilibrate the system and run the initial NVE production simulation.
- `job.sh`: An example SLURM job script to run the workflow on a compute cluster.
- `plan.py`: The workflow plan for StepUp.
- `output/`: Directory where all output files are saved.
- `utils.py`: Python script with utility functions.

## File Content Details

Most files contain comments and docstrings that explain their content in detail.
The NPZ files in the `output/` directory con100tain the following data:

- `time`: Time in seconds
- `potential_energy`: Potential energy in kJ/mol
- `kinetic_energy`: Kinetic energy in kJ/mol
- `total_energy`: Total energy in kJ/mol
- `temperature`: Temperature in Kelvin
- `volume`: Volume in m^3
- `atnums`: Atomic numbers
- `dipole`: Dipole moment in C.m

Quantities needed for the conductivity calculation are stored in base SI units
to avoid unit confusion in the worked example.

## Data Generation

All OpenMM simulations can be executed efficiently on a single compute node with StepUp:

```bash
OPENMM_CPU_THREADS=1 stepup boot -n 1.0
```

This will run in parallel the Jupyter notebooks `initial.ipynb` and `extension.ipynb`
multiple times with the correct arguments and in the right order.
All notebook outputs will be saved in the `output` directory.

Because StepUp already executes the workflow in parallel, multithreading in OpenMM is disabled.
The inputs were tested with OpenMM 8.2.0.

Note that the calculations take a while and require a lot of disk space.
You most likely want to run the workflow on a compute cluster.
Only the NPZ files in the `output` directory are used
as inputs for the worked example in the STACIE documentation.
