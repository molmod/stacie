#!/usr/bin/env python3
"""StepUp Plan to create ZIP files for Zenodo after all simulations are done."""

from path import Path
from stepup.core.api import copy, glob, mkdir, runpy, runsh, static
from stepup.reprep.api import compile_typst, make_inventory, sync_zenodo, zip_inventory

# Convert JupyText files to Jupyter notebooks
static("../source/", "../source/examples/", "../source/examples/matplotlibrc", "preprocess.py")
mkdir("preprocessed/")
copy("../source/examples/matplotlibrc", "./")
paths_ipynb = ["./matplotlibrc"]
for path_py in glob("../source/examples/*.py"):
    if path_py.name == "utils.py":
        copy(path_py, "./")
        paths_ipynb.append(path_py.name)
    else:
        path_pre = Path("preprocessed") / path_py.name
        runpy("./${inp} ${out}", inp=["preprocess.py", path_py], out=path_pre)
        path_ipynb = path_py.name.replace(".py", ".ipynb")
        paths_ipynb.append(path_ipynb)
        runsh("jupytext --to notebook ${inp} --output ${out}", inp=path_pre, out=path_ipynb)

# LAMMPS LJ3D example
static("zenodo.yaml", "zenodo.md")
static("lammps_lj3d/")
static("lammps_lj3d/sims/")
glob("lammps_lj3d/sims/replica_????_part_??/")
lammps_paths_txt = glob("lammps_lj3d/sims/replica_????_part_??/nve_*.txt")
lammps_paths_yaml = glob("lammps_lj3d/sims/replica_????_part_??/info.yaml")

# OpenMM Molten Salt example
static("openmm_salt/", "openmm_salt/output/")
openmm_paths_npz = glob("openmm_salt/output/*.npz")

# Compile the README

static("README.typ")
compile_typst("README.typ")

# Create inventory and zip file
paths = [
    *paths_ipynb,
    *lammps_paths_txt,
    *lammps_paths_yaml,
    *openmm_paths_npz,
]
make_inventory(*paths, "inventory.txt")
zip_inventory("inventory.txt", "examples.zip")
sync_zenodo("zenodo.yaml")
