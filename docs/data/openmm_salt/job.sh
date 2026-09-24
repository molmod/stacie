#!/usr/bin/env bash
#SBATCH --job-name openmm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=20G

export OPENMM_CPU_THREADS=1
time uv run sb
