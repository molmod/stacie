# LAMMPS simulations

**Tip:** The [Lammps Syntax Highlighting](https://marketplace.visualstudio.com/items?itemName=ThFriedrich.lammps)
greatly facilitates understanding and authoring LAMMPS input files.

All LAMMPS simulations can be executed efficiently on a single compute node with StepUp.
You can install StepUp as follows:

```
pip install stepup stepup-reprep
```

It is also assume that the LAMMPS executable is `lmp`.
The inputs were tested with version 29 Aug 2024, Update 1.

After all required software is installed,
you can rerun all the LAMMPS simulations as follows:

```bash
stepup
```
