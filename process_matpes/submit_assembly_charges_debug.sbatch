#!/bin/bash
# Debug-QOS smoke test for assemble_charges_xyz.py: processes a small --limit
# of npz files into a scratch --out-dir (never the real
# matpes_processed/xyz/ output) so it can run alongside the full production
# job (submit_assembly_charges.sbatch) without clobbering its output.
#
# Submit from the process_matpes directory:
#   sbatch submit_assembly_charges_debug.sbatch

#SBATCH --job-name=matpes-assemble-charges-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --account=matgen
#SBATCH --output=logs/assemble_charges_debug_%j.out

set -euo pipefail
mkdir -p logs

OUT_DIR="${PSCRATCH}/matpes_debug_test"
mkdir -p "${OUT_DIR}"

conda run --live-stream -n dft python -u -m matpes_pipeline.assemble_charges_xyz \
    --workers 32 \
    --limit 20000 \
    --out-dir "${OUT_DIR}"
