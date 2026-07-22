#!/bin/bash
# Build the HDF5 k-point/Fourier sidecar (see build_kspace_h5.py docstring).
# 32 workers shuttling decompressed npz arrays through multiprocessing IPC
# is too memory-heavy for the shared login node (OOM-killed there at
# 200k/774k files) -- needs a real compute-node memory allocation.
#
# Submit from the process_matpes directory:
#   sbatch submit_kspace_h5.sbatch
#
#SBATCH --job-name=matpes-kspace-h5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=04:00:00
#SBATCH --account=matgen
#SBATCH --output=logs/kspace_h5_%j.out

set -euo pipefail
mkdir -p logs

conda run --live-stream -n dft python -u -m matpes_pipeline.build_kspace_h5 --workers 32
