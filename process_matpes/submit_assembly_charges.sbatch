#!/bin/bash
# Stage 2b: join npz files with MatPES jsonl metadata, write the -charges
# extxyz (forces/stress/energy/DDEC6/formal-charges/multipoles; no Fourier).
#
# Submit from the process_matpes directory:
#   sbatch submit_assembly_charges.sbatch
#
#SBATCH --job-name=matpes-assemble-charges
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=06:00:00
#SBATCH --account=matgen
#SBATCH --output=logs/assemble_charges_%j.out

set -euo pipefail
mkdir -p logs

conda run --live-stream -n dft python -u -m matpes_pipeline.assemble_charges_xyz --workers 32
