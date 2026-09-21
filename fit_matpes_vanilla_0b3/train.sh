#!/bin/bash
#SBATCH --job-name=matpes_vanilla_0b3_debug
#SBATCH --constraint=gpu
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
#SBATCH --output=fit_matpes_vanilla_0b3/logs/%x_%j.out
#SBATCH --error=fit_matpes_vanilla_0b3/logs/%x_%j.err

# Vanilla ScaleShiftMACE on the MACE-MatPES (0b3) backbone: the no-electrostatics control.
#
#   sbatch fit_matpes_vanilla_0b3/train.sh                       # 8-node debug run (debug caps at 8 nodes / 30 min)
#   sbatch --job-name=matpes_vanilla_0b3 \
#          --qos=regular --time=12:00:00 \
#          fit_matpes_vanilla_0b3/train.sh                       # production, full 64-shard train set
#
# 8 nodes x 4 ranks = 32 ranks. --batch_size is PER RANK, so the effective batch is
# 16 x 32 = 512, against the ~128-256 of the 0b3 reference fit. config.yaml carries a
# matching 3x on lr. Changing --nodes without revisiting both is how the two stop lining up.
#
# --ema_decay is 0.9998, not the 0b3 value of 0.99999. Validation runs under the EMA
# weights (model_training_wrappers.py:145), and at 32 ranks x batch 16 this set is ~709
# steps/epoch, so the schedule's 90 epochs are only ~64k steps. 0.99999 is a 100k-step
# horizon -- longer than the whole run, leaving the eval weights about half converged at
# the end. 0.9998 is ~5.7k steps, about 8 epochs. Rescale it with the rank count.
#
# --pair_repulsion and --scheduler_patience 5 track 0b3.sh. ZBL needed a code change as
# well as the flag: build_model's model_config never carried pair_repulsion, so the flag
# alone was inert (see run_train_utils.py). Kept because every other axis here sits on the
# 0b3 side -- density interaction blocks, no edge_irreps, num_radial_basis 10 -- and MatPES
# is periodic inorganic data like MPtrj. The OMol-era models (omol.sh, MACE-POLAR-1) drop
# ZBL and Agnesi together, but they also change block, loss, optimizer and dataset at once.
#
# Still not tracked: 0b3's 'universal' loss (conditional Huber; every _LOSS_FUNCTIONS entry
# here is MSE) and --distance_transform Agnesi. The loss stays pinned to
# fit_matpes_lsc_baseline on purpose, so the backbone is the only variable.
#
# 0b3's --keep_checkpoints/--save_all_checkpoints are deliberately NOT set: checkpoints
# land in $HOME, which had 6.7 GiB free against ~150 MB per checkpoint. Keeping every
# improving epoch over 90 epochs would blow the 40 GiB home quota mid-run. Point
# --checkpoints_dir at $PSCRATCH first if you want to keep the full history.
#
# TRAIN_DIR overrides the train set and must stay out of production runs: the
# baseline is only comparable to the data-augmentation fit on the same data.
#   sbatch --export=ALL,TRAIN_DIR fit_matpes_vanilla_0b3/train.sh # 8-shard subset, smoke tests only
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
    echo "      under that directory, not $ROOT/fit_matpes_vanilla_0b3/logs/" >&2
fi

# Absolute, so nothing downstream depends on the working directory.
W="$ROOT/fit_matpes_vanilla_0b3"
DATA="$ROOT/matpes_fit/processed_r2scan_split"
# Always the full training set.
TRAIN_DIR="${TRAIN_DIR:-$DATA/train}"
# Scale goes in the run name: --restart_latest resumes by tag, so a shared one would make
# the 8- and 32-GPU runs resume each other's checkpoints.
NAME="matpes_vanilla_0b3_g${SLURM_NTASKS}"
mkdir -p "$W/logs" "$W/checkpoints/$NAME" "$W/results/$NAME"

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
export WANDB_RUN_ID="$NAME"
export WANDB_RESUME=allow
mkdir -p "$WANDB_DIR"

# E0s, atomic_numbers and avg_num_neighbors are deliberately not passed: statistics.json supplies them.
srun --cpu-bind=cores conda run -n mace_scf --no-capture-output python scripts/run_train.py \
    --name "$NAME" \
    --config "$W/config.yaml" \
    --work_dir "$W" \
    --log_dir "$W/logs" \
    --model_dir "$W" \
    --checkpoints_dir "$W/checkpoints/$NAME" \
    --results_dir "$W/results/$NAME" \
    --train_file "$TRAIN_DIR" \
    --valid_file "$DATA/val" \
    --statistics_file "$DATA/statistics.json" \
    --model ScaleShiftMACE \
    --scaling rms_forces_scaling \
    --hidden_irreps '128x0e + 128x1o' \
    --interaction_first RealAgnosticDensityInteractionBlock \
    --interaction RealAgnosticDensityResidualInteractionBlock \
    --num_interactions 2 \
    --correlation 3 \
    --max_ell 3 \
    --num_radial_basis 10 \
    --pair_repulsion \
    --MLP_irreps 16x0e \
    --weight_decay 1e-8 \
    --r_max 6.0 \
    --batch_size 16 \
    --valid_batch_size 32 \
    --num_workers 8 \
    --eval_interval 1 \
    --error_table PerAtomRMSEstressvirials \
    --optimizer adam \
    --amsgrad \
    --scheduler_patience 5 \
    --ema \
    --ema_decay 0.9998 \
    --compute_polarizability False \
    --compute_stress True \
    --default_dtype float64 \
    --enable_cueq True \
    --device cuda \
    --distributed \
    --seed 1 \
    --clip_grad 100 \
    --patience 40 \
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
