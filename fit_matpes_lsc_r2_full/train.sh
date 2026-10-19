#!/bin/bash
#SBATCH --job-name=lsc_r2_full
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=premium
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --account=matgen_g
#SBATCH --open-mode=append
#SBATCH --output=fit_matpes_lsc_r2_full/logs/%x_%j.out
#SBATCH --error=fit_matpes_lsc_r2_full/logs/%x_%j.err

# Full-data LocalSplitCharges fits on the R2 backbone, one case per job:
#
#   sbatch --export=ALL,CASE=control fit_matpes_lsc_r2_full/train.sh   # fit a: no multipole loss
#   sbatch --export=ALL,CASE=hub_10  fit_matpes_lsc_r2_full/train.sh   # fit b: + Huber multipoles, weight 10
#
# OX picks the SumOxidationStateMixer init: `zero` (default, --oxidation_state_mixer_zero_init),
# `ratio=<r>` (--oxidation_state_mixer_scale_ratio r: embedding RMS r x the element
# embedding's at init), or `default` (neither flag: the unscaled init, RMS ~6x the element
# embedding's, which blows up the first layer's activations). It goes into the run name, e.g. lsc_r2_oxratio1.0_control_g32.
#   sbatch --export=ALL,CASE=control,OX=ratio=1.0 fit_matpes_lsc_r2_full/train.sh
#
# FQ picks relabelled formal charges (process_matpes/matpes_pipeline/relabel_formal_charges.py):
# unset uses the original split; `zero` / `ddec` use processed_pbe_random_frame_split_fq<FQ>, where
# structures whose formal-charge Coulomb energy exceeded 2 eV/atom got zero / DDEC6-fitted
# integer charges. It goes into the run name as _fq<FQ>.
#
# The reference is R3 (fit_matpes_polarm_pbe_rf_sf): R2's backbone with schedule-free lr 0.01,
# betas 0.9/0.98, warmup 2000, clip 1, stage 1 only (epochs 0-49), same data, 32 ranks x 16.
# Every flag below is R3's except the model: LocalSplitCharges with the electrostatics of
# ../fit_matpes_lsc_r2_mp_scan, --local_scale_shift (R3's ScaleShiftMACE scale/shift), and
# --oxidation_state_mixer_zero_init True, which removes the init blow-up (product1 activation
# RMS 66 -> 0.1, step-0 force loss 38 -> 0.15). R2 itself used Adam, which blew up on LSC.
#
# 30 min walltime (early-training comparison only, ~6-7 epochs). Full stage 1 would need ~3 h
# (subset epochs are ~1.4x R2's) and 6 h to be safe. Longer runs can be chained 30-min jobs
# (chain.sh): --restart_latest now restores the live weights and EMA state saved beside each
# checkpoint (mace_scf/utils/train.py save_resume_state) and continues at the next epoch, so a
# resubmit no longer restarts from the EMA weights. Checkpoints from before that change have
# no resume state and still resume the old way.
# Subset peak memory was 9.8 GB per GPU.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/scripts/run_train.py" ]; do
    ROOT="$(dirname "$ROOT")"
done
[ -f "$ROOT/scripts/run_train.py" ] || { echo "submit from inside the mace-scf checkout" >&2; exit 1; }
cd "$ROOT"

W="$ROOT/fit_matpes_lsc_r2_full"
CASE="${CASE:?set CASE=control|hub_10}"
[ -f "$W/configs/$CASE.yaml" ] || { echo "no config for case $CASE" >&2; exit 1; }
DATA="${PSCRATCH:?PSCRATCH is unset}/mace-scf/matpes_fit/processed_pbe_random_frame_split"
FQ="${FQ:-}"
FQ_TAG=""
if [ -n "$FQ" ]; then
    case "$FQ" in zero|ddec) ;; *) echo "FQ must be zero or ddec, got $FQ" >&2; exit 1 ;; esac
    DATA="${DATA}_fq${FQ}"
    FQ_TAG="_fq${FQ}"
    [ -f "$DATA/relabel_report.json" ] || { echo "no relabelled data at $DATA" >&2; exit 1; }
fi
TRAIN_DIR="${TRAIN_DIR:-$DATA/train}"   # smoke tests only
read -r -a EXTRA_ARGS <<< "${EXTRA:-}"   # extra run_train.py flags, e.g. EXTRA=--save_all_checkpoints (chain.sh)
OX="${OX:-zero}"
case "$OX" in
    zero)    OX_ARGS=(--oxidation_state_mixer_zero_init True); OX_TAG=oxzero ;;
    default) OX_ARGS=(); OX_TAG=oxdefault ;;
    ratio=*) OX_ARGS=(--oxidation_state_mixer_scale_ratio "${OX#ratio=}"); OX_TAG="oxratio${OX#ratio=}" ;;
    *)       echo "OX must be zero, default or ratio=<r>, got $OX" >&2; exit 1 ;;
esac
# NOISE=<sigma> adds --formal_charge_noise_sigma (zero-sum Gaussian noise on the training formal
# charges) and _noise<sigma> to the run name. NOISE_IN_RANGE=1 also passes
# --formal_charge_noise_in_range True (noise shrunk per configuration so no charge leaves
# --oxidation_state_range) and names the run _noise<sigma>in.
NOISE="${NOISE:-}"
NOISE_ARGS=(); NOISE_TAG=""
if [ -n "$NOISE" ]; then NOISE_ARGS=(--formal_charge_noise_sigma "$NOISE"); NOISE_TAG="_noise${NOISE}"; fi
if [ -n "${NOISE_IN_RANGE:-}" ]; then
    [ -n "$NOISE" ] || { echo "NOISE_IN_RANGE needs NOISE" >&2; exit 1; }
    NOISE_ARGS+=(--formal_charge_noise_in_range True); NOISE_TAG="${NOISE_TAG}in"
fi
NAME="lsc_r2_${OX_TAG}${FQ_TAG}_${CASE}${NOISE_TAG}_g${SLURM_NTASKS}"
CKPT="${PSCRATCH}/mace-scf/fit_matpes_lsc_r2_full/checkpoints/$NAME"

# Outputs live on $SCRATCH: logs/results/wandb/models are links into $ROOT/fits (a link to
# $SCRATCH/mace-scf/fits), created here if missing. Each job also keeps a copy of the exact
# script it ran ($0 is SLURM's spooled copy), its config and its settings.
for sub in logs results wandb models; do
    if [ ! -e "$W/$sub" ]; then
        mkdir -p "$ROOT/fits/$(basename "$W")/$sub"
        ln -s "../fits/$(basename "$W")/$sub" "$W/$sub"
    fi
done
mkdir -p "$W/logs" "$CKPT" "$W/results/$NAME"
RECIPE="$W/logs/recipes/${NAME}_${SLURM_JOB_ID}"
mkdir -p "$RECIPE"
cp "$0" "$RECIPE/train.sh"
cp "$W/configs/$CASE.yaml" "$RECIPE/config.yaml"
printf 'CASE=%s\nOX=%s\nFQ=%s\nNOISE=%s\nNOISE_IN_RANGE=%s\nEXTRA=%s\nDATA=%s\ngit=%s\n' "$CASE" "$OX" "$FQ" "$NOISE" "${NOISE_IN_RANGE:-}" "${EXTRA:-}" "$DATA" \
    "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)$(git -C "$ROOT" diff --quiet 2>/dev/null || echo ' (dirty)')" > "$RECIPE/settings.txt"

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()' 2>/dev/null \
    || echo $((20000 + SLURM_JOB_ID % 40000)))
export MASTER_PORT
export OMP_NUM_THREADS=8
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export WANDB_MODE=offline
export WANDB_DIR="$W/wandb"
export WANDB_RUN_ID="$NAME"
export WANDB_RESUME=allow
mkdir -p "$WANDB_DIR"

srun --cpu-bind=cores conda run -n mace_scf --no-capture-output python scripts/run_train.py \
    --name "$NAME" \
    --config "$W/configs/$CASE.yaml" \
    --work_dir "$W" \
    --log_dir "$W/logs" \
    --model_dir "$W/models" \
    --checkpoints_dir "$CKPT" \
    --results_dir "$W/results/$NAME" \
    --train_file "$TRAIN_DIR" \
    --valid_file "$DATA/val" \
    --statistics_file "$DATA/statistics.json" \
    --model LocalSplitCharges \
    --local_scale_shift True \
    --scaling rms_forces_scaling \
    --formal_charges_from_data \
    --oxidation_state_range "(-8.0, 8.0)" \
    "${OX_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    "${NOISE_ARGS[@]}" \
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
    --batch_size 16 \
    --valid_batch_size 32 \
    --num_workers 8 \
    --eval_interval 1 \
    --error_table PerAtomMAERMSEstress \
    --patience 40 \
    --optimizer schedulefree \
    --beta 0.9 \
    --beta_two 0.98 \
    --warmup_steps_schedulefree 2000 \
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
    --debug-log-grad-summary \
    --wandb \
    --wandb_project matpes-lsc-comparison \
    --wandb_name "$NAME" \
    --wandb_dir "$WANDB_DIR" \
    --wandb_log_hypers lr batch_size r_max weight_decay max_num_epochs \
    --wandb-watch gradients \
    --wandb-watch-log-freq 25 \
    --restart_latest \
    --save_cpu
