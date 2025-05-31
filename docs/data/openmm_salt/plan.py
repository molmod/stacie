#!/usr/bin/env python3
from stepup.core.api import mkdir, runpy, static
from stepup.reprep.api import execute_papermill


def plan_initial(seed: int):
    """Run initial euqilibrium and production runs."""
    execute_papermill(
        "initial.ipynb",
        f"output/sim{seed:04d}_part00.ipynb",
        parameters={"seed": seed},
    )


def plan_extension(seed: int, part: int, nstep: int):
    """Extend the production run."""
    execute_papermill(
        "extension.ipynb",
        f"output/sim{seed:04d}_part{part:02d}.ipynb",
        # Let StepUp know that this notebook depends on the previous checkpoint.
        # This is also specified (with the amend function) in the notebook,
        # but by providing the dependency here,
        # StepUp knows about it before it executes the notebook,
        # which avoids unnecessary rescheduling.
        inp=f"output/sim{seed:04d}_part{part - 1:02d}_nve_last.chk",
        parameters={"seed": seed, "part": part, "nstep_nve": nstep},
    )


def plan_extract(seed: int, part: int, ensemble: str):
    """Extract essentials from OpenMM output (DCD and CSV) and store as NPZ."""
    runpy(
        "./${inp} ${out}",
        inp=[
            "extract.py",
            f"output/sim{seed:04d}_part{part:02d}_{ensemble}_traj.csv",
            f"output/sim{seed:04d}_part{part:02d}_{ensemble}_traj.dcd",
            f"output/sim{seed:04d}_part{part:02d}_{ensemble}_last.pdb",
        ],
        out=f"output/sim{seed:04d}_part{part:02d}_{ensemble}_traj.npz",
    )


mkdir("output")
static("bhmtf.py", "initial.ipynb", "extension.ipynb", "utils.py", "extract.py")
for seed in range(100):
    # Initial production runs
    plan_initial(seed)
    plan_extract(seed, 0, "nvt")
    plan_extract(seed, 0, "npt")
    plan_extract(seed, 0, "nve")

    # First extension of the production runs
    plan_extension(seed, 1, 8000)
    plan_extract(seed, 1, "nve")

    # Second extension of the production runs
    plan_extension(seed, 2, 184000)
    plan_extract(seed, 2, "nve")
