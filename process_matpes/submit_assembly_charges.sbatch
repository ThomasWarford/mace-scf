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
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=03:00:00
#SBATCH --account=matgen
#SBATCH --output=logs/assemble_charges_%j.out

set -euo pipefail
mkdir -p logs

# 128 hyperthreads = 64 physical cores (half a Perlmutter CPU node) under the
# shared QOS. This is embarrassingly parallel work (independent npz files
# chunked across an mp.Pool, no cross-worker synchronization until the final
# concatenation), so doubling workers from the original 32 roughly halves
# wall time for about the same total node-hours; --time is derated from 6h
# accordingly, with headroom.
conda run --live-stream -n dft python -u -m matpes_pipeline.assemble_charges_xyz --workers 64
