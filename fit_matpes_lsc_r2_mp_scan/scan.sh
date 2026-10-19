#!/bin/bash
#SBATCH --job-name=lsc_r2_mp_scan
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=debug
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --account=matgen_g
#SBATCH --output=fit_matpes_lsc_r2_mp_scan/logs/%x_%j.out
#SBATCH --error=fit_matpes_lsc_r2_mp_scan/logs/%x_%j.err

# Short scan to choose the multipole loss (form and weight) for the R2-backbone
# LocalSplitCharges fits. Four cases per job, each on its own 2 nodes / 8 GPUs, all run
# side by side until the 30 min debug limit; the per-epoch validation lines are the result.
#
#   sbatch --export=ALL,CASES="control l2_1 l2_10 l2_100" fit_matpes_lsc_r2_mp_scan/scan.sh
#   sbatch --export=ALL,CASES="l2_1000 hub_10 hub_100 hub_1000" fit_matpes_lsc_r2_mp_scan/scan.sh
#
# Cases are configs/<case>.yaml, which differ only in the multipole loss term. Everything
# else is the planned production run's (R2 backbone + fit_matpes_lsc_baseline's
# electrostatics), except the data: a 4-shard train / 4-shard val subset. The ranking
# therefore comes from early training and is a guide, not the answer.
#
# OPTIM picks the optimizer (cases as in ../fit_matpes_lsc_r2_opt_scan/scan.sh):
#   adam_ams     the original scan: Adam + AMSGrad, clip 100, lr 0.005 (default, for the old logs)
#   adam_clip1   Adam without AMSGrad, clip 1, lr 0.005
#   sf_clip1     schedule-free, lr 0.01, betas 0.9/0.98, warmup 200, clip 1
# The case configs carry lr 0.005; a per-job copy with OPTIM's lr goes to $SCR/configs.
# Logs are named lsc_r2_scan_<OPTIM>_<case> (adam_ams keeps the old lsc_r2_scan_<case>).
#
# EXTRA is appended to every case's run_train.py arguments, and TAG (required with EXTRA)
# goes into the log names as lsc_r2_scan_<OPTIM>_<TAG>_<case>, e.g.
#   sbatch --export=ALL,OPTIM=sf_clip1,TAG=oxzero,EXTRA="--oxidation_state_mixer_zero_init True",CASES="control hub_10" ...
#
# Checkpoints go to $PSCRATCH and are thrown away: --no-restart_latest.

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT"
[ -f scripts/run_train.py ] || { echo "submit from the mace-scf root" >&2; exit 1; }

W="$ROOT/fit_matpes_lsc_r2_mp_scan"
DATA="${PSCRATCH:?}/mace-scf/matpes_fit/processed_pbe_random_frame_split_sub4"
SCR="${PSCRATCH}/mace-scf/lsc_r2_mp_scan/${SLURM_JOB_ID}"
mkdir -p "$W/logs" "$SCR"
export OMP_NUM_THREADS=8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=disabled

read -r -a CASE_LIST <<< "${CASES:?set CASES}"
[ "${#CASE_LIST[@]}" -le 4 ] || { echo "at most 4 cases (2 nodes each)" >&2; exit 1; }

OPTIM="${OPTIM:-adam_ams}"
declare -A OPT=(
  [adam_ams]="--optimizer adam --amsgrad --clip_grad 100"
  [adam_clip1]="--optimizer adam --no-amsgrad --clip_grad 1"
  [sf_clip1]="--optimizer schedulefree --beta 0.9 --beta_two 0.98 --warmup_steps_schedulefree 200 --clip_grad 1"
)
declare -A LR=([adam_ams]=0.005 [adam_clip1]=0.005 [sf_clip1]=0.01)
[ -n "${OPT[$OPTIM]:-}" ] || { echo "unknown OPTIM $OPTIM" >&2; exit 1; }
PREFIX=lsc_r2_scan_; [ "$OPTIM" = adam_ams ] || PREFIX="lsc_r2_scan_${OPTIM}_"
EXTRA="${EXTRA:-}"; TAG="${TAG:-}"
[ -z "$EXTRA" ] || [ -n "$TAG" ] || { echo "set TAG with EXTRA" >&2; exit 1; }
[ -z "$TAG" ] || PREFIX="${PREFIX}${TAG}_"
read -r -a EXTRA_ARGS <<< "$EXTRA"
mkdir -p "$SCR/configs" "$W/logs/recipes/$SLURM_JOB_ID"
# keep a copy of the exact script this job ran ($0 is SLURM's spooled copy) and its settings
cp "$0" "$W/logs/recipes/$SLURM_JOB_ID/scan.sh"
printf 'CASES=%s\nOPTIM=%s\nTAG=%s\nEXTRA=%s\n' "${CASES}" "$OPTIM" "$TAG" "$EXTRA" > "$W/logs/recipes/$SLURM_JOB_ID/settings.txt"

NL=RealAgnosticResidualNonLinearInteractionBlock
i=0
PIDS=()
for CASE in "${CASE_LIST[@]}"; do
  [ -f "$W/configs/$CASE.yaml" ] || { echo "no config for case $CASE" >&2; exit 1; }
  NAME="${PREFIX}${CASE}"
  sed "s/^    lr: 0.005$/    lr: ${LR[$OPTIM]}/" "$W/configs/$CASE.yaml" > "$SCR/configs/$CASE.yaml"
  grep -q "^    lr: ${LR[$OPTIM]}$" "$SCR/configs/$CASE.yaml" || { echo "lr not set in $CASE" >&2; exit 1; }
  cp "$SCR/configs/$CASE.yaml" "$W/logs/recipes/$SLURM_JOB_ID/$CASE.yaml"
  # MACE's slurm launcher takes MASTER_ADDR from the first node of SLURM_JOB_NODELIST,
  # which is outside every step but the first. Each step's tasks point it at their own
  # step's nodes instead; ports differ per step so steps can never collide.
  srun --exact --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 \
      --cpus-per-task=32 --gpu-bind=none --cpu-bind=cores \
      --output="$W/logs/${NAME}_${SLURM_JOB_ID}.out" \
      bash -c 'export SLURM_JOB_NODELIST="$SLURM_STEP_NODELIST" MASTER_PORT='"$((29500 + i))"'; exec "$@"' _ \
      conda run -n mace_scf --no-capture-output python scripts/run_train.py \
      --name "$NAME" --config "$SCR/configs/$CASE.yaml" \
      --work_dir "$SCR/$CASE" --log_dir "$W/logs" --model_dir "$SCR/$CASE" \
      --checkpoints_dir "$SCR/$CASE/checkpoints" --results_dir "$SCR/$CASE/results" \
      --train_file "$DATA/train" --valid_file "$DATA/val" --statistics_file "$DATA/statistics.json" \
      --model LocalSplitCharges --local_scale_shift True \
      --formal_charges_from_data --oxidation_state_range "(-8.0, 8.0)" \
      --electrostatic_pbc_method pbc \
      --atomic_multipoles_max_l 1 --atomic_multipoles_smearing_width 1.5 --kspace_cutoff_factor 1.25 \
      --hidden_irreps '512x0e + 512x1o' --edge_irreps '128x0e + 128x1o' --use_agnostic_product True \
      --interaction_first $NL --interaction $NL \
      --num_interactions 2 --correlation 3 --max_ell 3 --num_radial_basis 8 --r_max 6.0 \
      --pair_repulsion --distance_transform Agnesi --MLP_irreps 16x0e \
      --weight_decay 1e-8 --batch_size 16 --valid_batch_size 32 --num_workers 8 \
      --eval_interval 1 --error_table PerAtomMAERMSEstress \
      ${OPT[$OPTIM]} --ema --ema_decay 0.9998 \
      --compute_polarizability False --compute_stress True --default_dtype float64 \
      --enable_cueq True --device cuda --distributed --seed 1 \
      --debug-log-grad-summary \
      --no-restart_latest --save_cpu "${EXTRA_ARGS[@]}" &
  PIDS+=($!)
  i=$((i + 1))
done
# peak memory on the batch node's 4 GPUs (the first case's first node)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -lms 5000 > "$W/logs/mem_${SLURM_JOB_ID}.csv" &
SMI=$!
for pid in "${PIDS[@]}"; do wait "$pid" || echo "a case exited non-zero (pid $pid)"; done
kill $SMI 2>/dev/null || true
