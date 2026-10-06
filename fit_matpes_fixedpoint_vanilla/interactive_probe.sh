#!/bin/bash
# Run one single-GPU probe through train.sh inside an interactive allocation, e.g.
#   salloc -N1 -C "gpu&hbm80g" -q interactive -t 00:30:00 -A matgen_g --ntasks-per-node=1 \
#       --gpus-per-node=1 --cpus-per-task=32 fit_matpes_fixedpoint_vanilla/interactive_probe.sh
# with the probe's settings (CASE, PROBE_SCRIPT, ...) in the environment.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
SLURM_NTASKS=1 SRUN_ARGS="-n1 --gpus=1 --exact" PROBE=1 bash fit_matpes_fixedpoint_vanilla/train.sh
