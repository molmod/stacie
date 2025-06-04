#!/usr/bin/env python3
"""Extract essentials from OpenMM output (DCD and CSV) and store as NPZ."""

import argparse

import mdtraj as md
import numpy as np
from path import Path


def main():
    args = parse_args()
    thermo = np.loadtxt(args.csv, skiprows=1, delimiter=",")
    traj = md.load(args.dcd, top=args.pdb)
    atnums = np.array([a.element.number for a in traj.top.atoms])
    atcoords = traj.xyz * 1e-9
    charges = (2 * (atnums == 11) - 1) * 1.602176634e-19
    np.savez_compressed(
        args.npz,
        time=thermo[:, 0] * 1e-12,
        potential_energy=thermo[:, 1],  # in kJ/mol
        kinetic_energy=thermo[:, 2],  # in kJ/mol
        total_energy=thermo[:, 3],  # in kJ/mol
        temperature=thermo[:, 4],
        volume=thermo[:, 5] * 1e-27,
        atnums=atnums,
        dipole=np.einsum("ijk,j->ki", atcoords, charges),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract essentials from OpenMM output (DCD and CSV) and store as NPZ."
    )
    parser.add_argument("csv", type=Path, help="Input CSV file")
    parser.add_argument("dcd", type=Path, help="Input DCD file", nargs="?")
    parser.add_argument("pdb", type=Path, help="Input PDB file", nargs="?")
    parser.add_argument("npz", type=Path, help="Output NPZ file")
    return parser.parse_args()


if __name__ == "__main__":
    main()
