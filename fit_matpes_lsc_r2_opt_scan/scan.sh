#!/bin/bash
#SBATCH --job-name=lsc_r2_opt_scan
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=debug
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --account=matgen_g
#SBATCH --output=fit_matpes_lsc_r2_opt_scan/logs/%x_%j.out
#SBATCH --error=fit_matpes_lsc_r2_opt_scan/logs/%x_%j.err

# Optimizer scan (job A) for the R2-backbone LocalSplitCharges fits, no multipole loss.
# The multipole-loss scan (../fit_matpes_lsc_r2_mp_scan) trained with AMSGrad, whose max of
# the second moment kept the clipped 1e10-1e12 epoch-0 gradients: its checkpoints show a
# median 17x cut in the effective lr (R2: 1.2x). These cases remove or bound that.
#
#   sbatch --export=ALL,CASES="adam_clip100 adam_clip5 adam_clip1 sf_clip1" fit_matpes_lsc_r2_opt_scan/scan.sh
#
#   adam_clip100  Adam without AMSGrad, clip 100 (the one-variable control for the others)
#   adam_clip5    Adam without AMSGrad, clip 5
#   adam_clip1    Adam without AMSGrad, clip 1
#   sf_clip1      schedule-free, Polar's recipe: lr 0.01, betas 0.9/0.98, clip 1; warmup 200
#                 steps, Polar's 2000 scaled to this ~5800-step debug run
#   sf_clip1_bs64 sf_clip1 at batch 64/GPU (global 512, Polar's), warmup 50 for its ~1450 steps;
#                 a single case, so submit it with --nodes=2:
#                 sbatch --nodes=2 --export=ALL,CASES=sf_clip1_bs64 fit_matpes_lsc_r2_opt_scan/scan.sh
#
# Batch 16/GPU as in every fit so far (global 128 here on 8 GPUs, 512 in production), with
# the multipole scan's lr 0.005 so the cases compare directly with it.
# Everything else is the multipole scan's: R2 backbone, fit_matpes_lsc_baseline's
# electrostatics, 4-shard train / 4-shard val subset, EMA 0.9998.
#
# Checkpoints go to $PSCRATCH and are thrown away: --no-restart_latest.

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT"
[ -f scripts/run_train.py ] || { echo "submit from the mace-scf root" >&2; exit 1; }

W="$ROOT/fit_matpes_lsc_r2_opt_scan"
DATA="${PSCRATCH:?}/mace-scf/matpes_fit/processed_pbe_random_frame_split_sub4"
SCR="${PSCRATCH}/mace-scf/lsc_r2_opt_scan/${SLURM_JOB_ID}"
mkdir -p "$W/logs" "$SCR"
export OMP_NUM_THREADS=8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=disabled

read -r -a CASE_LIST <<< "${CASES:?set CASES}"
[ "${#CASE_LIST[@]}" -le 4 ] || { echo "at most 4 cases (2 nodes each)" >&2; exit 1; }

NL=RealAgnosticResidualNonLinearInteractionBlock
declare -A CONFIG=([adam_clip100]=adam [adam_clip5]=adam [adam_clip1]=adam [sf_clip1]=schedulefree [sf_clip1_bs64]=schedulefree)
declare -A BATCH=([sf_clip1_bs64]=64)
declare -A OPT=(
  [adam_clip100]="--optimizer adam --no-amsgrad --clip_grad 100"
  [adam_clip5]="--optimizer adam --no-amsgrad --clip_grad 5"
  [adam_clip1]="--optimizer adam --no-amsgrad --clip_grad 1"
  [sf_clip1]="--optimizer schedulefree --beta 0.9 --beta_two 0.98 --warmup_steps_schedulefree 200 --clip_grad 1"
  [sf_clip1_bs64]="--optimizer schedulefree --beta 0.9 --beta_two 0.98 --warmup_steps_schedulefree 50 --clip_grad 1"
)
i=0
PIDS=()
for CASE in "${CASE_LIST[@]}"; do
  [ -n "${OPT[$CASE]:-}" ] || { echo "unknown case $CASE" >&2; exit 1; }
  NAME="lsc_r2_opt_${CASE}"
  # MACE's slurm launcher takes MASTER_ADDR from the first node of SLURM_JOB_NODELIST,
  # which is outside every step but the first. Each step's tasks point it at their own
  # step's nodes instead; ports differ per step so steps can never collide.
  srun --exact --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 \
      --cpus-per-task=32 --gpu-bind=none --cpu-bind=cores \
      --output="$W/logs/${NAME}_${SLURM_JOB_ID}.out" \
      bash -c 'export SLURM_JOB_NODELIST="$SLURM_STEP_NODELIST" MASTER_PORT='"$((29500 + i))"'; exec "$@"' _ \
      conda run -n mace_scf --no-capture-output python scripts/run_train.py \
      --name "$NAME" --config "$W/configs/${CONFIG[$CASE]}.yaml" \
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
      --weight_decay 1e-8 --batch_size "${BATCH[$CASE]:-16}" --valid_batch_size 32 --num_workers 8 \
      --eval_interval 1 --error_table PerAtomMAERMSEstress \
      ${OPT[$CASE]} --ema --ema_decay 0.9998 \
      --compute_polarizability False --compute_stress True --default_dtype float64 \
      --enable_cueq True --device cuda --distributed --seed 1 \
      --debug-log-grad-summary \
      --no-restart_latest --save_cpu &
  PIDS+=($!)
  i=$((i + 1))
done
# peak memory on the batch node's 4 GPUs (the first case's first node)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -lms 5000 > "$W/logs/mem_${SLURM_JOB_ID}.csv" &
SMI=$!
for pid in "${PIDS[@]}"; do wait "$pid" || echo "a case exited non-zero (pid $pid)"; done
kill $SMI 2>/dev/null || true
