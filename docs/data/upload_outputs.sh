#!/usr/bin/env bash
if [ -z "$1" ]; then
    echo "Error: Remote host argument missing."
    exit 1
fi
rsync -av --info=progress2 \
    lammps_lj3d openmm_salt \
    --include=lammps_lj3d/replica_????_part_??/*.yaml \
    --include=lammps_lj3d/replica_????_part_??/nve_*.txt \
    --include=openmm_salt/output/*.npz \
    --exclude=*.* \
    --prune-empty-dirs \
    $1:projects/emd-viscosity/stacie/
