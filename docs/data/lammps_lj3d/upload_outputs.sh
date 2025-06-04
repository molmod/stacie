#!/usr/bin/env bash
if [ -z "$1" ]; then
    echo "Error: Remote host argument missing."
    exit 1
fi
rsync -av --info=progress2 \
    sims \
    --include=sims/replica_????_part_??/*.yaml \
    --include=sims/replica_????_part_??/nv?_*.txt \
    --exclude=*.* \
    --prune-empty-dirs \
    $1:projects/emd-viscosity/stacie/lammps_lj3d/
