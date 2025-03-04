#!/usr/bin/env python3
from runlammps import runlammps
from stepup.core.api import mkdir, static
from stepup.reprep.api import render_jinja

static("runlammps.py", "exploration/", "exploration/in.lammps")
runlammps("exploration/")

static("template-prod-init.lammps", "template-prod-extend.lammps")
mkdir("production/")
nreplica = 100
for ireplica in range(nreplica):
    name = f"production/replica_{ireplica:04d}_part_00"
    mkdir(f"{name}/")
    render_jinja("template-prod-init.lammps", {"seed": ireplica + 1}, f"{name}/in.lammps")
    runlammps(f"{name}/", inp=["exploration/nve_final.restart"])

# Number of steps to use in the extensions of the production runs
extend_num_steps = [4000]
for ireplica in range(nreplica):
    for iextend, num_steps in enumerate(extend_num_steps):
        name = f"production/replica_{ireplica:04d}_part_{iextend + 1:02d}"
        mkdir(f"{name}/")
        render_jinja(
            "template-prod-extend.lammps",
            {
                "previous_dir": f"../replica_{ireplica:04d}_part_{iextend:02d}",
                "additional_steps": num_steps,
            },
            f"{name}/in.lammps",
        )
        runlammps(
            f"{name}/",
            inp=[f"production/replica_{ireplica:04d}_part_{iextend:02d}/nve_final.restart"],
        )
