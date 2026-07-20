#!/bin/bash
# Stage 2: join npz files with MatPES jsonl metadata and write extxyz.
# Runs in the shared QOS: 32 workers only need a quarter of a node, so a
# 64-hyperthread (32 physical core) slice is charged instead of a full node.
# The jsonl index is a few GB and is shared with workers via fork.
#
# Submit from the process_matpes directory:
#   sbatch submit_assembly.sbatch
#
#SBATCH --job-name=matpes-assemble
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=06:00:00
#SBATCH --account=matgen
#SBATCH --output=logs/assemble_%j.out

set -euo pipefail
mkdir -p logs

conda run --live-stream -n dft python -u -m matpes_pipeline.assemble_xyz --workers 32
