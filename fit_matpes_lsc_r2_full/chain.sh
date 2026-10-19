#!/bin/bash
# Run one train.sh case to a target epoch as a chain of 30-min debug-queue jobs, each resuming
# the last with --restart_latest (exact since the resume-state change; see train.sh). Runs on
# the login node, e.g. in the background:
#
#   fit_matpes_lsc_r2_full/chain.sh <CASE> <FQ> <TARGET_EPOCH> [NOISE]
#   fit_matpes_lsc_r2_full/chain.sh mse_5 zero 40
#   fit_matpes_lsc_r2_full/chain.sh mse_5 zero 40 0.5    # + formal charge noise sigma 0.5
#   NOISE_IN_RANGE=1 fit_matpes_lsc_r2_full/chain.sh hub_5 zero 40 0.5   # ... kept in range
#
# OX is ratio=1.0. Every job adds --save_all_checkpoints, so a job whose epochs don't lower the
# validation loss still leaves a checkpoint to resume from (~300 MB each, on $SCRATCH). The chain
# stops when the run log shows TARGET_EPOCH, when a job ends other than by hitting its walltime
# (a crash), or when two jobs in a row add no epoch. Progress goes to logs/chain_<run name>.log.

set -euo pipefail
CASE=${1:?CASE}; FQ=${2:?FQ}; TARGET=${3:?TARGET_EPOCH}; NOISE=${4:-}
NOISE_TAG=${NOISE:+_noise$NOISE}
# NOISE_IN_RANGE=1 in the environment reaches train.sh through --export=ALL (noise kept in
# --oxidation_state_range; run name _noise<sigma>in)
[ -n "${NOISE_IN_RANGE:-}" ] && NOISE_TAG="${NOISE_TAG}in"
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
W="$ROOT/fit_matpes_lsc_r2_full"
NAME="lsc_r2_oxratio1.0_fq${FQ}_${CASE}${NOISE_TAG}_g32"   # as train.sh names it
JOB="lsc_r2_full_r1_fq${FQ}_${CASE}${NOISE_TAG}"
RUNLOG="$W/logs/${NAME}_run-1.log"
CHAINLOG="$W/logs/chain_${NAME}.log"
cd "$ROOT"

last_epoch() { { grep -ho "Epoch [0-9]*:" "$RUNLOG" 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1; } || true; }
say() { echo "$(date '+%F %T') $*" | tee -a "$CHAINLOG"; }

stalls=0
while :; do
    before=$(last_epoch); before=${before:--1}
    if [ "$before" -ge "$TARGET" ]; then say "done: epoch $before >= $TARGET"; exit 0; fi
    # debug allows 5 queued jobs per user; wait for room rather than fail
    until jid=$(sbatch --parsable --qos=debug --time=00:30:00 --job-name="$JOB" \
        --export=ALL,CASE="$CASE",OX=ratio=1.0,FQ="$FQ",NOISE="$NOISE",EXTRA=--save_all_checkpoints \
        "$W/train.sh" 2>>"$CHAINLOG"); do sleep 120; done
    say "submitted $jid at epoch $before"
    while [ -n "$(squeue -h -j "$jid" -o %i 2>/dev/null)" ]; do sleep 60; done
    sleep 30
    after=$(last_epoch); after=${after:--1}
    state=$(sacct -j "$jid" -X -n -o State | xargs)
    say "job $jid $state $(sacct -j "$jid" -X -n -o Elapsed | xargs): epoch $before -> $after"
    # a healthy link hits the walltime (stage 1 ends at epoch 99); a killed job prints
    # tracebacks too, so the SLURM state is the signal, not the .err
    if [ "$state" != "TIMEOUT" ] && [ "$after" -lt "$TARGET" ]; then
        say "stop: job $jid ended $state before the target, see logs/${JOB}_${jid}.err"; exit 1
    fi
    if [ "$after" -le "$before" ]; then
        stalls=$((stalls + 1))
        [ "$stalls" -ge 2 ] && { say "stop: two jobs without a new epoch"; exit 1; }
    else
        stalls=0
    fi
done
