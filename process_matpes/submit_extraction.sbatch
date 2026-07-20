#!/bin/bash
# Stage 1: extract all MatPES launcher dirs to npz, as a SLURM array job.
# Each array task is one full CPU node processing an interleaved shard.
# Resubmitting is safe: existing npz files are skipped.
#
# Submit from the process_matpes directory:
#   sbatch submit_extraction.sbatch
#
#SBATCH --job-name=matpes-extract
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --account=matgen
#SBATCH --requeue
#SBATCH --output=logs/extract_%A_%a.out

set -euo pipefail
mkdir -p logs

# One shard per array task; SLURM provides the task count, so the array range
# above (or `sbatch --array=0-N` at submit time) is the only thing to change.
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-8}

conda run --live-stream -n dft python -u -m matpes_pipeline.run_extraction \
    --num-shards "$NUM_SHARDS" \
    --shard-id "$SLURM_ARRAY_TASK_ID" \
    --workers 128
