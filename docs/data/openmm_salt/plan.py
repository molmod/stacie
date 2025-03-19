#!/usr/bin/env python3
from stepup.core.api import mkdir, static
from stepup.reprep.api import convert_jupyter

mkdir("output")
static("bhmtf.py", "exploration.ipynb", "production.ipynb", "utils.py")
convert_jupyter("exploration.ipynb", "output/exploration.html")
for seed in range(100):
    convert_jupyter(
        "production.ipynb",
        f"output/prod{seed:04d}.html",
        # Let StepUp know that this notebook uses a PDB file is input.
        # This could be done with amend in the notebook, but this would cause
        # a lot of rescheduling, making the workflow less efficient.
        # By providing the dependency here, StepUp knows this dependency
        # before it executes the notebook.
        inp="output/exploration_nve_final.pdb",
        nbargs={"seed": seed},
    )
