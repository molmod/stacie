#!/usr/bin/env bash
if [ -z "$1" ]; then
    echo "Error: Remote host argument missing."
    exit 1
fi
rsync -av --delete --info=progress2 $1:projects/emd-viscosity/stacie/openmm_salt/output .
