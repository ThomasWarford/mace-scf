#!/bin/bash
# preprocess_random_frame_split.sh (PBE only) plus a fermi_level property, for the FixedPoint
# fits (fit_matpes_fixedpoint_vanilla). The existing processed_pbe_random_frame_split has no
# fermi_level: its heads never mapped one, and new_atomic_data.py then silently fills 0.0, but
# FixedPoint's direct mode feeds the reference Fermi level to the model as an input.
#
# Two VASP Fermi-level conventions, one array task each, otherwise identical:
#   task 0 (ab):  REF_vasp_fermi_level_plus_alpha_bet  -> processed_pbe_random_frame_split_fermi_ab
#   task 1 (raw): REF_vasp_fermi_level (raw E-fermi)    -> processed_pbe_random_frame_split_fermi_raw
# (see process_matpes/README.md for the variants). preprocess_data.py writes the training-set
# fermi_level_offset into statistics.json.
#
# Filters, E0s, valid fraction and the (hardcoded 1234) split seed match
# preprocess_random_frame_split.sh, so the valid frames are the same as the LSC fits'.
#
# Submit from the mace-scf repository root:
#   sbatch matpes_fit/preprocess_random_frame_split_fermi.sh

#SBATCH --job-name=matpes-preprocess-fermi
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=01:30:00
#SBATCH --account=matgen
#SBATCH --output=matpes_fit/logs/preprocess_fermi_%A_%a.out
#SBATCH --error=matpes_fit/logs/preprocess_fermi_%A_%a.err

set -euo pipefail
mkdir -p matpes_fit/logs

TAGS=(ab raw)
FERMI_KEYS=(REF_vasp_fermi_level_plus_alpha_bet REF_vasp_fermi_level)
tag="${TAGS[$SLURM_ARRAY_TASK_ID]}"
fermi_key="${FERMI_KEYS[$SLURM_ARRAY_TASK_ID]}"

XYZ_DIR=/global/cfs/cdirs/matgen/esoteric/matpes_processed/xyz
NUM_PROCESS=64
R_MAX=6.0
SEED=123
VALID_FRACTION=0.03

train_file="${XYZ_DIR}/MatPES-PBE-charges-quick.xyz"
# $PSCRATCH, not $HOME (see preprocess_random_frame_split.sh): ~3.7 GB per task.
OUT_ROOT="${PSCRATCH:?PSCRATCH is unset}/mace-scf/matpes_fit"
h5_prefix="${OUT_ROOT}/processed_pbe_random_frame_split_fermi_${tag}/"
mkdir -p "${h5_prefix}"

e0s=$(cat e0s_matpes_pbe.txt)
heads='{"default": {"info_keys": {"energy": "REF_energy", "total_charge": "REF_total_charge", "stress": "REF_stress", "fermi_level": "'"${fermi_key}"'"}, "arrays_keys": {"forces": "REF_forces", "charges": "REF_formal_charges", "atomic_multipoles": "REF_multipoles"}}}'
echo "fermi_level <- ${fermi_key}; output ${h5_prefix}"

conda run --live-stream -n mace_scf python -u scripts/preprocess_data.py \
    --train_file="${train_file}" \
    --valid_fraction="${VALID_FRACTION}" \
    --h5_prefix="${h5_prefix}" \
    --work_dir="${h5_prefix}" \
    --r_max="${R_MAX}" \
    --E0s="${e0s}" \
    --heads="${heads}" \
    --compute_statistics \
    --num_process="${NUM_PROCESS}" \
    --seed="${SEED}" \
    --allow_low_density_pbc \
    --max_force=20.0 \
    --require_finite_multipoles \
    --shuffle=True
