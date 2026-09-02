#!/usr/bin/env bash
#SBATCH --job-name lammps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

export OMP_NUM_THREADS=1
time uv run sb
