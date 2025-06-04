#!/usr/bin/env bash
#SBATCH --job-name openmm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

cd ../../../../../
. activate
cd ${SLURM_SUBMIT_DIR}
export OPENMM_CPU_THREADS=1
# export OPENMM_DEFAULT_PLATFORM=CPU
stepup boot -n ${SLURM_CPUS_PER_TASK}
