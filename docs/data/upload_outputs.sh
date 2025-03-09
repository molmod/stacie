#!/usr/bin/env bash
if [ -z "$1" ]; then
    echo "Error: Remote host argument missing."
    exit 1
fi
rsync -avR --info=progress2 \
    lammps_lj3d/exploration/*.txt\
    lammps_lj3d/exploration/*.yaml \
    lammps_lj3d/exploration/*.png \
    lammps_lj3d/produciton/replica_*/*.txt \
    lammps_lj3d/produciton/replica_*/*.yaml \
    lammps_lj3d/produciton/replica_*/*.png \
    openmm_salt/output/*.* \
    $1:projects/emd-viscosity/stacie/
