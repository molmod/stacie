#!/usr/bin/env python3
from path import Path
from runlammps import runlammps
from stepup.core.api import render_jinja, static


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
    old = f"replica_{ireplica:04d}_part_{part - 1:02d}"
    olddir = Path(f"sims/{old}/")
    new = f"replica_{ireplica:04d}_part_{part:02d}"
    newdir = Path(f"sims/{new}/")
    render_jinja(
        "template-ext.lammps",
        {
            "previous_dir": f"../{old}",
            "additional_steps": additional_steps,
        },
        newdir / "in.lammps",
    )
    runlammps(
        newdir,
        inp=[olddir / "nve_final.restart"],
        out=[newdir / "nve_final.restart"],
    )
    return newdir


static("runlammps.py", "template-init.lammps", "template-ext.lammps")
nreplica = 100
for ireplica in range(nreplica):
    # Initial production run
    name_i = f"sims/replica_{ireplica:04d}_part_00"
    render_jinja("template-init.lammps", {"seed": ireplica + 1}, f"{name_i}/in.lammps")
    runlammps(f"{name_i}/", out=[f"{name_i}/nve_final.restart"])

    # Extensions of the production run
    plan_extension(ireplica=ireplica, part=1, additional_steps=24000)
    plan_extension(ireplica=ireplica, part=2, additional_steps=64000)
