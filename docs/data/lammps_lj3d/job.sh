#!/usr/bin/env bash
#SBATCH --job-name lammps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

cd ../../../../../
. activate
cd ${SLURM_SUBMIT_DIR}
time stepup boot -n ${SLURM_CPUS_PER_TASK}
