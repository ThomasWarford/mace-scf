#!/bin/bash
# Preprocess MatPES-{PBE,R2SCAN}-charges.xyz into sharded HDF5 for on-the-fly
# dataloading, via scripts/preprocess_data.py.
#
# Submit from the mace-scf repository root:
#   sbatch matpes_fit/preprocess.sh
#
# NOTE: the -charges.xyz files must come from the current version of
# process_matpes/matpes_pipeline/assemble_charges_xyz.py (which reuses
# assemble_xyz.py's build_atoms and so includes REF_forces/REF_stress/
# REF_total_charge alongside REF_formal_charges/REF_multipoles). Files
# generated before that update lack forces/stress/total_charge. Regenerate
# with `sbatch submit_assembly_charges.sbatch` from process_matpes/ first if
# unsure.
#
# scripts/preprocess_data.py reads the whole --train_file with ASE in a
# single process before any sharding/multiprocessing happens, so this step
# is not parallelized across NUM_PROCESS. The -charges.xyz files are ~1.5GB
# (Fourier/k-space arrays already stripped), much lighter than the raw
# 177GB MatPES-PBE.xyz, so this should complete quickly.
#
#SBATCH --job-name=matpes-preprocess
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=02:00:00
#SBATCH --account=matgen
#SBATCH --output=matpes_fit/logs/preprocess_%j.out

set -euo pipefail
mkdir -p matpes_fit/logs

XYZ_DIR=/global/cfs/cdirs/matgen/esoteric/matpes_processed/xyz
NUM_PROCESS=64
R_MAX=6.0
VALID_FRACTION=0.01
SEED=123

# arrays_keys.charges -> REF_formal_charges (pymatgen-guessed oxidation
# states), not REF_ddec6_charges (DDEC6 partial charges); arrays_keys.
# atomic_multipoles -> REF_multipoles (l<=1, DDEC6-derived, e3nn (y,z,x)
# order). REF_total_charge is always 0.0 in this data (neutral DFT cells).
run_preprocess () {
    local functional="$1"
    local train_file="${XYZ_DIR}/MatPES-${functional}-charges.xyz"
    local h5_prefix="matpes_fit/processed_${functional,,}/"
    local e0s
    e0s=$(cat "e0s_matpes_${functional,,}.txt")
    local heads
    heads=$(cat <<EOF
{"default": {"info_keys": {"energy": "REF_energy", "total_charge": "REF_total_charge", "stress": "REF_stress"}, "arrays_keys": {"forces": "REF_forces", "charges": "REF_formal_charges", "atomic_multipoles": "REF_multipoles"}}}
EOF
    )

    conda run --live-stream -n mace_scf python -u scripts/preprocess_data.py \
        --train_file="${train_file}" \
        --valid_fraction="${VALID_FRACTION}" \
        --h5_prefix="${h5_prefix}" \
        --r_max="${R_MAX}" \
        --E0s="${e0s}" \
        --heads="${heads}" \
        --compute_statistics \
        --num_process="${NUM_PROCESS}" \
        --seed="${SEED}" \
        --shuffle=True
}

run_preprocess PBE
run_preprocess R2SCAN
