#!/bin/bash
#SBATCH --job-name=lsc_r2_lrfind
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=debug
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --account=matgen_g
#SBATCH --output=fit_matpes_lsc_r2_full/logs/%x_%j.out
#SBATCH --error=fit_matpes_lsc_r2_full/logs/%x_%j.err

# LR range test (--lr_find, mace_scf/utils/lr_find.py) for train.sh's best stage-1 recipe:
# CASE=hub_5, OX=ratio=1.0, FQ=zero, from the initialised model. Same flags as train.sh except
# 16 ranks x 32 = the same global batch of 512 as 32 x 16, a fresh name/checkpoint dir with
# --no-restart_latest (never the trained hub_5 checkpoint), no wandb, and the lr_find flags.
# Schedule-free is replaced by AdamW with the same betas for the test (run_train.py logs it).
#
#   sbatch fit_matpes_lsc_r2_full/lr_find.sh
#   sbatch --export=ALL,CASE=control fit_matpes_lsc_r2_full/lr_find.sh
#
# Outputs: results/lrfind/<name>/<name>_run-1_stage1_lrfind_r<k>.{png,csv,json}.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/scripts/run_train.py" ]; do
    ROOT="$(dirname "$ROOT")"
done
[ -f "$ROOT/scripts/run_train.py" ] || { echo "submit from inside the mace-scf checkout" >&2; exit 1; }
cd "$ROOT"

W="$ROOT/fit_matpes_lsc_r2_full"
CASE="${CASE:-hub_5}"
[ -f "$W/configs/$CASE.yaml" ] || { echo "no config for case $CASE" >&2; exit 1; }
DATA="${PSCRATCH:?PSCRATCH is unset}/mace-scf/matpes_fit/processed_pbe_random_frame_split_fqzero"
STEPS="${STEPS:-500}"
REPEATS="${REPEATS:-2}"
NAME="lsc_r2_oxratio1.0_fqzero_${CASE}_lrfind_g${SLURM_NTASKS}"
CKPT="${PSCRATCH}/mace-scf/fit_matpes_lsc_r2_full/checkpoints/$NAME"
OUT="$W/results/lrfind/$NAME"

mkdir -p "$W/logs" "$CKPT" "$OUT"
RECIPE="$W/logs/recipes/${NAME}_${SLURM_JOB_ID}"
mkdir -p "$RECIPE"
cp "$0" "$RECIPE/lr_find.sh"
cp "$W/configs/$CASE.yaml" "$RECIPE/config.yaml"
printf 'CASE=%s\nSTEPS=%s\nREPEATS=%s\nDATA=%s\ngit=%s\n' "$CASE" "$STEPS" "$REPEATS" "$DATA" \
    "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)$(git -C "$ROOT" diff --quiet 2>/dev/null || echo ' (dirty)')" > "$RECIPE/settings.txt"

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()' 2>/dev/null \
    || echo $((20000 + SLURM_JOB_ID % 40000)))
export MASTER_PORT
export OMP_NUM_THREADS=8
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

srun --cpu-bind=cores conda run -n mace_scf --no-capture-output python scripts/run_train.py \
    --name "$NAME" \
    --config "$W/configs/$CASE.yaml" \
    --work_dir "$W" \
    --log_dir "$W/logs" \
    --model_dir "$OUT" \
    --checkpoints_dir "$CKPT" \
    --results_dir "$OUT" \
    --train_file "$DATA/train" \
    --valid_file "$DATA/val" \
    --statistics_file "$DATA/statistics.json" \
    --model LocalSplitCharges \
    --local_scale_shift True \
    --scaling rms_forces_scaling \
    --formal_charges_from_data \
    --oxidation_state_range "(-8.0, 8.0)" \
    --oxidation_state_mixer_scale_ratio 1.0 \
    --electrostatic_pbc_method pbc \
    --atomic_multipoles_max_l 1 \
    --atomic_multipoles_smearing_width 1.5 \
    --kspace_cutoff_factor 1.25 \
    --hidden_irreps '512x0e + 512x1o' \
    --edge_irreps '128x0e + 128x1o' \
    --use_agnostic_product True \
    --interaction_first RealAgnosticResidualNonLinearInteractionBlock \
    --interaction RealAgnosticResidualNonLinearInteractionBlock \
    --num_interactions 2 \
    --correlation 3 \
    --max_ell 3 \
    --num_radial_basis 8 \
    --pair_repulsion \
    --distance_transform Agnesi \
    --MLP_irreps 16x0e \
    --weight_decay 1e-8 \
    --r_max 6.0 \
    --batch_size 32 \
    --valid_batch_size 32 \
    --num_workers 8 \
    --eval_interval 1 \
    --error_table PerAtomMAERMSEstress \
    --optimizer schedulefree \
    --beta 0.9 \
    --beta_two 0.98 \
    --ema \
    --ema_decay 0.9998 \
    --compute_polarizability False \
    --compute_stress True \
    --default_dtype float64 \
    --enable_cueq True \
    --device cuda \
    --distributed \
    --seed 1 \
    --clip_grad 1.0 \
    --no-restart_latest \
    --save_cpu \
    --lr_find \
    --lr_find_min 1e-5 \
    --lr_find_max 1.0 \
    --lr_find_steps "$STEPS" \
    --lr_find_repeats "$REPEATS"
