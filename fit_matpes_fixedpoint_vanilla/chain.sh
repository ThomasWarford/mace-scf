#!/bin/bash
# Run one train.sh case to a target epoch as a chain of 30-min debug-queue jobs, each resuming
# the last with --restart_latest (adapted from ../fit_matpes_lsc_r2_full/chain.sh). Runs on the
# login node, e.g. in the background:
#
#   fit_matpes_fixedpoint_vanilla/chain.sh <CASE> <TARGET_EPOCH> [FERMI]
#   fit_matpes_fixedpoint_vanilla/chain.sh vanilla 59          # FERMI=ab
#   fit_matpes_fixedpoint_vanilla/chain.sh vanilla 59 raw
#
# Every job adds --save_all_checkpoints, so a job whose epochs don't lower the validation loss
# still leaves a checkpoint to resume from. The stages switch inside a job when one ends, so one
# chain covers direct, linearize_solve and unroll_scf. The chain stops when the run log shows
# TARGET_EPOCH, when a job ends other than by hitting its walltime (a crash), or when two jobs in
# a row add no epoch. Progress goes to logs/chain_<run name>.log.

set -euo pipefail
CASE=${1:?CASE}; TARGET=${2:?TARGET_EPOCH}; FERMI=${3:-ab}
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
W="$ROOT/fit_matpes_fixedpoint_vanilla"
NAME="fp_vanilla_fermi${FERMI}_${CASE}_g32"   # as train.sh names it
JOB="fp_vanilla_fermi${FERMI}_${CASE}"
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
        --export=ALL,CASE="$CASE",FERMI="$FERMI",EXTRA=--save_all_checkpoints \
        "$W/train.sh" 2>>"$CHAINLOG"); do sleep 120; done
    say "submitted $jid at epoch $before"
    while [ -n "$(squeue -h -j "$jid" -o %i 2>/dev/null)" ]; do sleep 60; done
    sleep 30
    after=$(last_epoch); after=${after:--1}
    state=$(sacct -j "$jid" -X -n -o State | xargs)
    say "job $jid $state $(sacct -j "$jid" -X -n -o Elapsed | xargs): epoch $before -> $after"
    # a healthy link hits the walltime; a killed job prints tracebacks too, so the SLURM state
    # is the signal, not the .err
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
