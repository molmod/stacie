#!/usr/bin/env python3
"""StepUp Plan to create ZIP files for Zenodo after all simulations are done."""

from path import Path
from stepup.core.api import copy, run, static
from stepup.reprep.api import compile_typst, make_inventory, sync_zenodo, zip_inventory

# Convert JupyText files to Jupyter notebooks
static("preprocess.py", "../source/examples/matplotlibrc")
copy("../source/examples/matplotlibrc", "./")
paths_ipynb = ["./matplotlibrc"]
for path_py in static("../source/examples/*.py"):
    if path_py.name == "utils.py":
        copy(path_py, "./")
        paths_ipynb.append(path_py.name)
    else:
        path_pre = Path("preprocessed") / path_py.name
        run(f"./preprocess.py {path_py} {path_pre}", inp=path_py, out=path_pre)
        path_ipynb = path_py.name.replace(".py", ".ipynb")
        paths_ipynb.append(path_ipynb)
        run(
            f"jupytext --to notebook {path_pre} --output {path_ipynb}", inp=path_pre, out=path_ipynb
        )

# LAMMPS LJ3D example
lammps_paths_txt = static("lammps_lj3d/sims/replica_????_part_??/nv?_*.txt")
lammps_paths_yaml = static("lammps_lj3d/sims/replica_????_part_??/info.yaml")

# OpenMM Molten Salt example
openmm_paths_npz = static("openmm_salt/output/*.npz")

# Cloud cover example data
static("cloud-cover/download.sh")
cloudcover_paths = static("cloud-cover/*.csv")

# Compile the README
static("README.typ")
compile_typst("README.typ")

# Create inventory and zip file
paths = [
    *paths_ipynb,
    *lammps_paths_txt,
    *lammps_paths_yaml,
    *openmm_paths_npz,
    *cloudcover_paths,
]
make_inventory(*paths, "inventory.txt")

# Sync with Zenodo
static("sync_zenodo.yaml", "zenodo_description.md")
zip_inventory("inventory.txt", "examples.zip")
sync_zenodo(
    "sync_zenodo.yaml", ["README.pdf", "examples.zip"], path_description="zenodo_description.md"
)
