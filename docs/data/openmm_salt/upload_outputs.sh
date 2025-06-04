#!/usr/bin/env bash
if [ -z "$1" ]; then
    echo "Error: Remote host argument missing."
    exit 1
fi
rsync -av --info=progress2 \
    output \
    --include=output/*.npz \
    --exclude=*.* \
    --prune-empty-dirs \
    $1:projects/emd-viscosity/stacie/openmm_salt/
