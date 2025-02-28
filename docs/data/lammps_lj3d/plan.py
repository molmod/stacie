#!/usr/bin/env python3
from stepup.core.api import mkdir, static, step
from stepup.reprep.api import render_jinja


def run_lammps(workdir, inp=(), out=()):
    """Build rule for LAMMPS"""
    # This implementation is incomplete because it does not track all outputs.
    # You may have to clean them up manually with `git clean -dfX .`
    inp = ["in.lammps", *inp]
    out = ["log.txt", *out]
    step("lmp -i in.lammps -l log.txt -sc none", inp=inp, out=out, workdir=workdir)


static("exploration/", "exploration/in.lammps")
run_lammps("exploration/", out=["nve_final.restart"])

static("template.lammps")
mkdir("production/")
for irep in range(100):
    name = f"production/{irep:04d}"
    mkdir(f"{name}/")
    render_jinja("template.lammps", {"seed": irep + 1}, f"{name}/in.lammps")
    run_lammps(f"{name}/", inp=["../../exploration/nve_final.restart"])
