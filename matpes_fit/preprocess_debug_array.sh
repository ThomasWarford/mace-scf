#!/bin/bash
# Debug-QOS array variant of preprocess.sh: one array task per functional
# (task 0 = PBE, task 1 = R2SCAN) instead of running both sequentially in
# one job, so each gets its own fresh 30-min debug allocation. Writes to a
# separate _debug output dir so it can't race with a concurrent
# preprocess.sh run using the real h5_prefix (e.g. the regular-QOS job
# already queued as a fallback).
#
# Submit from the mace-scf repository root:
#   sbatch matpes_fit/preprocess_debug_array.sh
#
# If a task doesn't finish inside 30 min, fall back to preprocess.sh on
# --qos=regular.

#SBATCH --job-name=matpes-preprocess-debug
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --account=matgen
#SBATCH --output=matpes_fit/logs/preprocess_debug_%A_%a.out

set -euo pipefail
mkdir -p matpes_fit/logs

FUNCTIONALS=(PBE R2SCAN)
functional="${FUNCTIONALS[$SLURM_ARRAY_TASK_ID]}"

XYZ_DIR=/global/cfs/cdirs/matgen/esoteric/matpes_processed/xyz
NUM_PROCESS=64
R_MAX=6.0
VALID_FRACTION=0.01
SEED=123

train_file="${XYZ_DIR}/MatPES-${functional}-charges-quick.xyz"
h5_prefix="matpes_fit/_processed_${functional,,}_debug/"
e0s=$(cat "e0s_matpes_${functional,,}.txt")
heads='{"default": {"info_keys": {"energy": "REF_energy", "total_charge": "REF_total_charge", "stress": "REF_stress"}, "arrays_keys": {"forces": "REF_forces", "charges": "REF_formal_charges", "atomic_multipoles": "REF_multipoles"}}}'

conda run --live-stream -n mace_scf python -u scripts/preprocess_data.py \
    --train_file="${train_file}" \
    --valid_fraction="${VALID_FRACTION}" \
    --h5_prefix="${h5_prefix}" \
    --r_max="${R_MAX}" \
    --E0s="${e0s}" \
    --heads="${heads}" \
    --compute_statistics \
    --num_process="${NUM_PROCESS}" \
    --seed="${SEED}" \
    --shuffle=True
