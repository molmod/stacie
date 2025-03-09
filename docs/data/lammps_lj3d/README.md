# LAMMPS 3D Lennard-Jones Simulations

**Tip:** The [Lammps Syntax Highlighting](https://marketplace.visualstudio.com/items?itemName=ThFriedrich.lammps)
greatly facilitates understanding and authoring LAMMPS input files.

All LAMMPS simulations can be executed efficiently on a single compute node with StepUp.
You can install StepUp as follows:

```bash
pip install stepup stepup-reprep
```

It is also assumed that the LAMMPS executable is `lmp`.
The inputs were tested with the LAMMPS release of 29 Aug 2024, Update 1.

After all required software is installed,
you can rerun all the LAMMPS simulations as follows:

```bash
stepup
```
