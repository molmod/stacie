#!/usr/bin/env python3
from runlammps import runlammps
from stepup.core.api import mkdir, render_jinja, static


def plan_extension(ireplica: int, part: int, additional_steps: int):
    """
    Plan the extension of a production run.

    Parameters
    ----------
    ireplica
        Replica index, different for each independent production run.
    part
        Part index, 1 for the first extension, 2 for the second, etc.
    additional_steps
        Number of additional steps to run in this extension.
    """
    name = f"sims/replica_{ireplica:04d}_part_{part:02d}"
    mkdir(f"{name}/")
    render_jinja(
        "template-ext.lammps",
        {
            "previous_dir": f"../replica_{ireplica:04d}_part_{part - 1:02d}",
            "additional_steps": additional_steps,
        },
        f"{name}/in.lammps",
    )
    runlammps(
        f"{name}/", inp=[f"sims/replica_{ireplica:04d}_part_{part - 1:02d}/nve_final.restart"]
    )
    return name


static("runlammps.py", "template-init.lammps", "template-ext.lammps")
mkdir("sims/")
nreplica = 100
for ireplica in range(nreplica):
    # Initial production run
    name_i = f"sims/replica_{ireplica:04d}_part_00"
    mkdir(f"{name_i}/")
    render_jinja("template-init.lammps", {"seed": ireplica + 1}, f"{name_i}/in.lammps")
    runlammps(f"{name_i}/")

    # Extensions of the production run
    plan_extension(ireplica=ireplica, part=1, additional_steps=24000)
    plan_extension(ireplica=ireplica, part=2, additional_steps=64000)
