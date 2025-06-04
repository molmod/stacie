#!/usr/bin/env python3
from runlammps import runlammps
from stepup.core.api import mkdir, render_jinja, static

static("runlammps.py", "template-init.lammps", "template-ext.lammps")
mkdir("sims/")
nreplica = 100
for ireplica in range(nreplica):
    # Initial production run
    name_i = f"sims/replica_{ireplica:04d}_part_00"
    mkdir(f"{name_i}/")
    render_jinja("template-init.lammps", {"seed": ireplica + 1}, f"{name_i}/in.lammps")
    runlammps(f"{name_i}/")

    # Extension 1 of the production run
    name_e1 = f"sims/replica_{ireplica:04d}_part_01"
    mkdir(f"{name_e1}/")
    render_jinja(
        "template-ext.lammps",
        {"previous_dir": f"../replica_{ireplica:04d}_part_00", "additional_steps": 24000},
        f"{name_e1}/in.lammps",
    )
    runlammps(f"{name_e1}/", inp=[f"{name_i}/nve_final.restart"])

    # Extension 2 of the production run
    name_e2 = f"sims/replica_{ireplica:04d}_part_02"
    mkdir(f"{name_e2}/")
    render_jinja(
        "template-ext.lammps",
        {"previous_dir": f"../replica_{ireplica:04d}_part_01", "additional_steps": 64000},
        f"{name_e2}/in.lammps",
    )
    runlammps(f"{name_e2}/", inp=[f"{name_e1}/nve_final.restart"])
