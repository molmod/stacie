#!/usr/bin/env python3
"""A simple wrapper for launching LAMMPS simulations.

See DESCRIPTION below for more information, or run this script with the --help
option.
"""

import argparse
import subprocess

from path import Path
from stepup.core.api import amend, runpy

__all__ = ("runlammps",)


def runlammps(workdir: str, inp: list[str] = ()):
    workdir = Path(workdir)
    runpy(
        f"./runlammps.py {workdir}",
        inp=["runlammps.py", workdir / "in.lammps", *inp],
        out=[workdir / "log.txt"],
    )


def main(argv: list[str] | None = None):
    """Main program."""
    args = parse_args(argv)
    for file in args.rundir.files():
        if file.name != "in.lammps":
            file.remove()
    subprocess.run(
        ["lmp", "-i", "in.lammps", "-l", "log.txt", "-sc", "none"],
        cwd=args.rundir,
        check=True,
    )
    extra_out = [path for path in args.rundir.files() if path.name not in ["in.lammps", "log.txt"]]
    amend(out=extra_out)


DESCRIPTION = """
Run a single LAMMPS calculation in a directory.

This script reduces the risk that left-overs from previous runs will interfere
with the current run. It also informs StepUp of all the output files that are
created by LAMMPS, which is difficult to know in advance.

It is assumed that there is a single input file, `in.lammps`, in a directory
where the simulation is to be run. All other files are considered output files,
and if they exist before starting the simulation, they are removed.
Subdirectories are ignored and not removed.

LAMMPS will be called as follows, suppressing screen output and writing the log
to log.txt:

lmp -i in.lammps -l log.txt -sc none

If you want to use this script in a StepUp plan.py script in the same directory,
you can use the following code:

from runlammps import runlammps
from stepup.core.api import static
static("runlammps.py", "exploration/", "exploration/in.lammps")
runlammps("exploration/")

You may provide additional input files, e.g. retart files with the `inp` option.
If your input file is created by a previous step, you don't need to declare it
as a static file.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="runlammps.py",
        description=DESCRIPTION,
    )
    parser.add_argument(
        "rundir", type=Path, help="The directory where the in.lammps file is located."
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
