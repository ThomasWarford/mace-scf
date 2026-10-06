#!/bin/bash
#SBATCH --job-name=fp_vanilla
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=debug
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --account=matgen_g
#SBATCH --open-mode=append
#SBATCH --output=fit_matpes_fixedpoint_vanilla/logs/%x_%j.out
#SBATCH --error=fit_matpes_fixedpoint_vanilla/logs/%x_%j.err

# First self-consistent (FixedPoint) fit to MatPES-PBE: the "vanilla" FixedPoint model (default
# OneBodyVariableUpdate / BiasedLinearPotentialEmbedding / NoNonLinearity update and
# StrictQuadraticFieldEnergyReadout) on a 128x0e+128x1o backbone (fit_matpes_vanilla's size),
# trained direct -> linearize_solve -> unroll_scf (configs/vanilla.yaml), as the docs' fixed-point
# training page recommends.
#
#   sbatch --export=ALL,CASE=vanilla fit_matpes_fixedpoint_vanilla/train.sh            # or chain.sh
#   sbatch --export=ALL,CASE=vanilla,FERMI=raw fit_matpes_fixedpoint_vanilla/train.sh
#
# FERMI picks the reference Fermi level (matpes_fit/preprocess_random_frame_split_fermi.sh):
# `ab` (default) = REF_vasp_fermi_level_plus_alpha_bet, `raw` = REF_vasp_fermi_level. Direct mode
# feeds it to the model as an input; the SCF stages use it as the initial guess.
#
# Differences from the LSC recipe (../fit_matpes_lsc_r2_full/train.sh), all forced by FixedPoint:
# no cueq / agnostic product / edge_irreps / local_scale_shift (each raises for FixedPoint), no
# formal charges or oxidation-state flags (ignored), and no stress (FixedPointCore returns none).
# Optimizer and electrostatics are R3's / LSC's.
#
# BATCH_SIZE is one value for all three stages, chosen by the worst-case memory probe
# (probe_batch_memory.py: the B largest train frames; MatPES runs to 243 atoms, median 5).
# linearize_solve's dense batch Jacobian dominates and grows ~n_atoms^2. Worst-case train step:
#   40 GB A100: B=4 linsolve solve OOMs and silently falls back to a 15-step unroll; B=2 OOMs in
#               the backward through the solve (job 59141897)
#   80 GB A100: B=2 fits every stage, linsolve 52.1 GB (66%), others <= 13 GB (job 59164804)
# hence hbm80g and B=2. Do not raise B without re-probing: an OOM inside the solve is caught by
# the wrapper and turns into an unroll fallback rather than an error.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/scripts/run_train.py" ]; do
    ROOT="$(dirname "$ROOT")"
done
[ -f "$ROOT/scripts/run_train.py" ] || { echo "submit from inside the mace-scf checkout" >&2; exit 1; }
cd "$ROOT"

W="$ROOT/fit_matpes_fixedpoint_vanilla"
CASE="${CASE:?set CASE=vanilla|smoke}"
[ -f "$W/configs/$CASE.yaml" ] || { echo "no config for case $CASE" >&2; exit 1; }
FERMI="${FERMI:-ab}"
case "$FERMI" in ab|raw) ;; *) echo "FERMI must be ab or raw, got $FERMI" >&2; exit 1 ;; esac
DATA="${PSCRATCH:?PSCRATCH is unset}/mace-scf/matpes_fit/processed_pbe_random_frame_split_fermi_${FERMI}"
# PROBE_DATA (memory probe only): any split with the same frames, e.g. the pre-Fermi one; memory
# does not depend on the Fermi values.
if [ -n "${PROBE_DATA:-}" ]; then
    [ -n "${PROBE:-}" ] || { echo "PROBE_DATA is for PROBE=1 only" >&2; exit 1; }
    DATA="$PROBE_DATA"
fi
[ -f "$DATA/statistics.json" ] || { echo "no preprocessed data at $DATA" >&2; exit 1; }
TRAIN_DIR="${TRAIN_DIR:-$DATA/train}"   # smoke tests only
VALID_DIR="${VALID_DIR:-$DATA/val}"     # smoke tests only (full-val linsolve eval is ~20 min on 4 GPUs)
# CUEQ=1: cuEquivariance backbone (--enable_cueq; the SCF field blocks stay e3nn). Refused for
# linearize_solve stages, so use it with configs that have none (e.g. smoke_cueq).
CUEQ_ARGS=(); CUEQ_TAG=""
if [ -n "${CUEQ:-}" ]; then CUEQ_ARGS=(--enable_cueq True); CUEQ_TAG="_cueq"; fi
# NO_EMA=1: no weight EMA (e.g. overfit tests, where a 0.9998 EMA lags ~5000 steps behind)
EMA_ARGS=(--ema --ema_decay 0.9998)
[ -n "${NO_EMA:-}" ] && EMA_ARGS=()
read -r -a EXTRA_ARGS <<< "${EXTRA:-}"   # extra run_train.py flags, e.g. EXTRA=--save_all_checkpoints (chain.sh)
BATCH_SIZE="${BATCH:-2}"   # probe_batch_memory.py, see above; BATCH=<n> overrides (direct/unroll
                           # with CUEQ=1 only; never for linearize_solve) and tags the run name _b<n>
BATCH_TAG=""; [ -n "${BATCH:-}" ] && BATCH_TAG="_b${BATCH}"
# CLIP=<x>: gradient clip (default 1.0, which clips ~every step under the docs' MSE weights);
# tags the run name _clip<x>
CLIP_GRAD="${CLIP:-1.0}"; CLIP_TAG=""; [ -n "${CLIP:-}" ] && CLIP_TAG="_clip${CLIP}"
# `average` over train_0.h5 (5.8k shuffled frames, FERMI=ab) in smoke job 59165031; hard-coded so
# each chain link skips a full-data pass at startup.
FIELD_NORMS="[16.85128938, 11.29358665, 0.45036201, 0.25651795]"
NAME="fp_vanilla_fermi${FERMI}_${CASE}${CUEQ_TAG}${BATCH_TAG}${CLIP_TAG}_g${SLURM_NTASKS}"
CKPT="${PSCRATCH}/mace-scf/fit_matpes_fixedpoint_vanilla/checkpoints/$NAME"

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
printf 'CLIP=%s\nBATCH=%s\nCUEQ=%s\nCASE=%s\nFERMI=%s\nTRAIN_DIR=%s\nVALID_DIR=%s\nEXTRA=%s\nDATA=%s\ngit=%s\n' "$CLIP_GRAD" "$BATCH_SIZE" "${CUEQ:-}" "$CASE" "$FERMI" "$TRAIN_DIR" "$VALID_DIR" "${EXTRA:-}" "$DATA" \
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

# PROBE=1 runs probe_batch_memory.py (worst-case memory per stage and batch size, one GPU) with
# exactly these flags instead of training: submit with -N1 -n1 (see that script).
PY_SCRIPT=scripts/run_train.py
DIST_ARGS=(--distributed)
if [ -n "${PROBE:-}" ]; then
    PY_SCRIPT="$W/${PROBE_SCRIPT:-probe_batch_memory.py}"   # or e.g. diag_init_gradients.py
    DIST_ARGS=()
    NAME="${NAME}_probe"
    mkdir -p "$W/results/$NAME"
fi

# E0s, atomic_numbers, avg_num_neighbors and fermi_level_offset come from statistics.json.
# SRUN_ARGS: extra srun flags, e.g. to pack several single-GPU runs into one interactive
# allocation: SLURM_NTASKS=1 SRUN_ARGS="-n1 --gpus=1 --exact" fit_matpes_fixedpoint_vanilla/train.sh
read -r -a SRUN_EXTRA <<< "${SRUN_ARGS:-}"
srun --cpu-bind=cores "${SRUN_EXTRA[@]}" conda run -n mace_scf --no-capture-output python "$PY_SCRIPT" \
    --name "$NAME" \
    --config "$W/configs/$CASE.yaml" \
    --work_dir "$W" \
    --log_dir "$W/logs" \
    --model_dir "$W/models" \
    --checkpoints_dir "$CKPT" \
    --results_dir "$W/results/$NAME" \
    --train_file "$TRAIN_DIR" \
    --valid_file "$VALID_DIR" \
    --statistics_file "$DATA/statistics.json" \
    --model FixedPoint \
    --hidden_irreps '128x0e + 128x1o' \
    --num_interactions 2 \
    --correlation 3 \
    --max_ell 3 \
    --MLP_irreps 16x0e \
    --r_max 6.0 \
    --electrostatic_pbc_method pbc \
    --atomic_multipoles_max_l 1 \
    --atomic_multipoles_smearing_width 1.5 \
    --kspace_cutoff_factor 1.25 \
    --field_feature_max_l 1 \
    --field_feature_widths "[1.5, 3.0]" \
    --field_feature_norms "$FIELD_NORMS" \
    --weight_decay 1e-8 \
    --batch_size "$BATCH_SIZE" \
    --valid_batch_size "$BATCH_SIZE" \
    --num_workers 8 \
    --eval_interval 1 \
    --error_table PerAtomMAERMSEstress \
    --patience 40 \
    --optimizer schedulefree \
    --beta 0.9 \
    --beta_two 0.98 \
    --warmup_steps_schedulefree 2000 \
    "${EMA_ARGS[@]}" \
    --compute_polarizability False \
    --compute_stress False \
    --default_dtype float64 \
    --device cuda \
    "${DIST_ARGS[@]}" \
    --seed 1 \
    --clip_grad "$CLIP_GRAD" \
    "${CUEQ_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
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
