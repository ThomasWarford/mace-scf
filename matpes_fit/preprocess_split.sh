#!/bin/bash
# Debug-QOS array variant of preprocess_debug_array.sh that takes an explicit,
# pre-computed train/valid split (process_matpes/matpes_pipeline/
# split_by_structure.py) instead of letting mace draw a random per-frame
# valid_fraction split -- avoids leaking near-duplicate structures (same
# mp-id, different ionic step / volume scaling) across train and valid.
# One array task per functional (task 0 = PBE, task 1 = R2SCAN).
#
# Submit from the mace-scf repository root:
#   sbatch matpes_fit/preprocess_split.sh

#SBATCH --job-name=matpes-preprocess-split
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --account=matgen
#SBATCH --output=matpes_fit/logs/preprocess_split_%A_%a.out

set -euo pipefail
mkdir -p matpes_fit/logs

FUNCTIONALS=(PBE R2SCAN)
functional="${FUNCTIONALS[$SLURM_ARRAY_TASK_ID]}"

XYZ_DIR=/global/cfs/cdirs/matgen/esoteric/matpes_processed/xyz
NUM_PROCESS=64
R_MAX=6.0
SEED=123

train_file="${XYZ_DIR}/MatPES-${functional}-charges-quick-train.xyz"
valid_file="${XYZ_DIR}/MatPES-${functional}-charges-quick-valid.xyz"
h5_prefix="matpes_fit/processed_${functional,,}_split/"
e0s=$(cat "e0s_matpes_${functional,,}.txt")
heads='{"default": {"info_keys": {"energy": "REF_energy", "total_charge": "REF_total_charge", "stress": "REF_stress"}, "arrays_keys": {"forces": "REF_forces", "charges": "REF_formal_charges", "atomic_multipoles": "REF_multipoles"}}}'

conda run --live-stream -n mace_scf python -u scripts/preprocess_data.py \
    --train_file="${train_file}" \
    --valid_file="${valid_file}" \
    --h5_prefix="${h5_prefix}" \
    --r_max="${R_MAX}" \
    --E0s="${e0s}" \
    --heads="${heads}" \
    --compute_statistics \
    --num_process="${NUM_PROCESS}" \
    --seed="${SEED}" \
    --shuffle=True
