#!/bin/bash
# One-off: repair npz files whose matpes_id is None/empty (matpes-mp-structures
# jobs; see matpes_pipeline/repair_matpes_ids.py). Shared QOS half-node slice.
#
# Submit from the process_matpes directory:
#   sbatch submit_repair.sbatch
#
#SBATCH --job-name=matpes-repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=04:00:00
#SBATCH --account=matgen
#SBATCH --output=logs/repair_%j.out

set -euo pipefail
mkdir -p logs

conda run --live-stream -n dft python -u -m matpes_pipeline.repair_matpes_ids --workers 64
