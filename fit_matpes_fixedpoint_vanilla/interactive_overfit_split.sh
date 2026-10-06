#!/bin/bash
# Run inside an interactive allocation (1 hbm80g node), from the repo root, e.g.
#   salloc -N1 -C "gpu&hbm80g" -q interactive -t 03:45:00 -A matgen_g --ntasks-per-node=4 \
#       --gpus-per-node=4 --cpus-per-task=32 fit_matpes_fixedpoint_vanilla/interactive_overfit_split.sh
# 1. density split (diag_init_gradients.py) on snapshots of the docs and D overfit checkpoints;
# 2. extend both overfit runs (docs all-L2 weights; D) in parallel, one GPU each, for HOURS;
# 3. density split again on their final checkpoints. Progress: logs/interactive/status_<job>.txt
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
W=fit_matpes_fixedpoint_vanilla
HOURS=${HOURS:-3}
L=$W/logs/interactive; ST=$L/status_${SLURM_JOB_ID}.txt
say() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
OV=$PSCRATCH/mace-scf/fit_matpes_fixedpoint_vanilla/overfit_data
C=$PSCRATCH/mace-scf/fit_matpes_fixedpoint_vanilla/checkpoints
SNAP=$PSCRATCH/mace-scf/fit_matpes_fixedpoint_vanilla/split_snapshots/$SLURM_JOB_ID; mkdir -p $SNAP
HS="--fixedpoint-initial-charge-head-scale 1.0"
one() { SLURM_NTASKS=1 SRUN_ARGS="-n1 --gpus=1 --cpus-per-task=32 --exact" TRAIN_DIR=$OV VALID_DIR=$OV "$@"; }
latest() { ls -t $C/fp_vanilla_fermiab_$1_g1/*_direct_epoch-*.pt | head -1; }
diag() {  # $1 case, $2 checkpoint, $3 tag, $4 extra flags
    one env CASE=$1 PROBE=1 PROBE_SCRIPT=diag_init_gradients.py DIAG_BATCHES=32 DIAG_CHECKPOINTS="$2" EXTRA="$4" \
        bash $W/train.sh > $L/diag_${3}_${SLURM_JOB_ID}.log 2>&1
    say "diag $3 exit $? ($(basename $2))"
}
say "start on $(hostname), HOURS=$HOURS"
cp "$(latest overfit_docs)" "$(latest overfit_hs1_noq_mse)" $SNAP/
diag overfit_docs "$SNAP/$(basename $(latest overfit_docs))" docs_before "" &
diag overfit_hs1_noq_mse "$SNAP/$(basename $(latest overfit_hs1_noq_mse))" D_before "$HS" &
wait
say "training: docs and D for ${HOURS}h"
one env CASE=overfit_docs timeout ${HOURS}h bash $W/train.sh > $L/train_docs_${SLURM_JOB_ID}.log 2>&1 &
one env CASE=overfit_hs1_noq_mse EXTRA="$HS" timeout ${HOURS}h bash $W/train.sh > $L/train_D_${SLURM_JOB_ID}.log 2>&1 &
wait
say "training done: docs at $(basename $(latest overfit_docs)), D at $(basename $(latest overfit_hs1_noq_mse))"
diag overfit_docs "$(latest overfit_docs)" docs_after "" &
diag overfit_hs1_noq_mse "$(latest overfit_hs1_noq_mse)" D_after "$HS" &
wait
say "DONE"
