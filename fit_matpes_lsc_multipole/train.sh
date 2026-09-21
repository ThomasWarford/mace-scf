#!/bin/bash
#SBATCH --job-name=matpes_lsc_multipole_debug
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --account=matgen_g
#SBATCH --open-mode=append
#SBATCH --output=fit_matpes_lsc_multipole/logs/%x_%j.out
#SBATCH --error=fit_matpes_lsc_multipole/logs/%x_%j.err

# LocalSplitCharges fit with DDEC6 multipoles in the loss, on the preprocessed MatPES R2SCAN shards.
#
#   sbatch fit_matpes_lsc_multipole/train.sh                       # 2-node debug smoke test
#   sbatch --job-name=matpes_lsc_multipole \
#          --qos=regular --time=12:00:00 \
#          fit_matpes_lsc_multipole/train.sh                       # production
#
# The job name is the W&B run id, so resubmissions under --restart_latest append to one run.

set -euo pipefail

# Find the repo root: the submit dir is not always it (job 58545264), and SLURM spools this script so BASH_SOURCE cannot be used.
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/scripts/run_train.py" ]; do
    ROOT="$(dirname "$ROOT")"
done
if [ ! -f "$ROOT/scripts/run_train.py" ]; then
    echo "could not find scripts/run_train.py above ${SLURM_SUBMIT_DIR:-$PWD}" >&2
    echo "submit this script from inside the mace-scf checkout" >&2
    exit 1
fi
cd "$ROOT"
echo "repo root: $ROOT"

# SLURM resolves #SBATCH --output/--error against the submit dir before this runs, so those two files follow it.
if [ "${SLURM_SUBMIT_DIR:-$ROOT}" != "$ROOT" ]; then
    echo "note: submitted from ${SLURM_SUBMIT_DIR}, so the SLURM .out/.err for this job are" >&2
    echo "      under that directory, not $ROOT/fit_matpes_lsc_multipole/logs/" >&2
fi

# Absolute, so nothing downstream depends on the working directory.
W="$ROOT/fit_matpes_lsc_multipole"
DATA="$ROOT/matpes_fit/processed_r2scan_split"
# Always the full training set; ~5.8 min/epoch on 8 GPUs, so two epochs fit the debug window.
TRAIN_DIR="${TRAIN_DIR:-$DATA/train}"
mkdir -p "$W/logs" "$W/checkpoints" "$W/results"

# Kernel-assigned free port on the batch host, which is the MASTER_ADDR nodelist[0] (fixed ports hit EADDRINUSE, job 58245266).
MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()' 2>/dev/null \
    || echo $((20000 + SLURM_JOB_ID % 40000)))
export MASTER_PORT
echo "MASTER_PORT=$MASTER_PORT"

export OMP_NUM_THREADS=8
# e3nn loads its constants with torch.load, which torch >= 2.6 refuses by default.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Compute nodes have no outbound network; sync offline runs with ion_conductivity/fits/warmup_check/wandb_import.py.
export WANDB_MODE=offline
export WANDB_DIR="$W/wandb"
export WANDB_RUN_ID="$SLURM_JOB_NAME"
export WANDB_RESUME=allow
mkdir -p "$WANDB_DIR"

# E0s, atomic_numbers and avg_num_neighbors are deliberately not passed: statistics.json supplies them.
srun --cpu-bind=cores conda run -n mace_scf --no-capture-output python scripts/run_train.py \
    --name matpes_lsc_multipole \
    --config "$W/config.yaml" \
    --work_dir "$W" \
    --log_dir "$W/logs" \
    --model_dir "$W" \
    --checkpoints_dir "$W/checkpoints" \
    --results_dir "$W/results" \
    --train_file "$TRAIN_DIR" \
    --valid_file "$DATA/val" \
    --statistics_file "$DATA/statistics.json" \
    --model LocalSplitCharges \
    --formal_charges_from_data \
    --oxidation_state_range "(-8.0, 8.0)" \
    --electrostatic_pbc_method pbc \
    --atomic_multipoles_max_l 1 \
    --atomic_multipoles_smearing_width 1.5 \
    --kspace_cutoff_factor 1.25 \
    --hidden_irreps '128x0e + 128x1o' \
    --r_max 6.0 \
    --batch_size 32 \
    --valid_batch_size 64 \
    --num_workers 8 \
    --eval_interval 1 \
    --error_table PerAtomRMSEstressvirials \
    --optimizer adam \
    --amsgrad \
    --ema \
    --ema_decay 0.99 \
    --compute_polarizability False \
    --compute_stress True \
    --default_dtype float64 \
    --enable_cueq True \
    --device cuda \
    --distributed \
    --seed 1 \
    --clip_grad 100 \
    --debug-log-grad-summary \
    --wandb \
    --wandb_project matpes-lsc-comparison \
    --wandb_name "$SLURM_JOB_NAME" \
    --wandb_dir "$WANDB_DIR" \
    --wandb_log_hypers lr batch_size r_max weight_decay max_num_epochs \
    --wandb-watch gradients \
    --wandb-watch-log-freq 25 \
    --restart_latest \
    --save_cpu
