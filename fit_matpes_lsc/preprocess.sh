#!/bin/bash
#SBATCH --job-name=matpes-preprocess
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --time=0:30:00
#SBATCH --account=matgen

#SBATCH --output=matpes_fit/logs/preprocess_%j.out
#SBATCH --error=matpes_fit/logs/preprocess_%j.err

set -euo pipefail
mkdir -p matpes_fit/logs

MATPES_XYZ=/global/cfs/cdirs/matgen/esoteric/matpes_processed/xyz/MatPES-PBE-charges.xyz
H5_PREFIX=matpes_fit/processed/
NUM_PROCESS=128
R_MAX=6.0
VALID_FRACTION=0.05
SEED=123

HEADS='{"default": {"info_keys": {"energy": "REF_energy", "stress": "REF_stress", "total_charge": "REF_total_charge", "dipole": "REF_dipole"}, "arrays_keys": {"forces": "REF_forces", "charges": "REF_formal_charges", "atomic_multipoles": "REF_multipoles"}}}'

# NOTE: keys must be given via --heads, not --energy_key/--forces_key/--stress_key --
# mace_scf.utils.check_args.check_and_fix_heads asserts every *_key flag is None and
# raises if it isn't (info_keys/arrays_keys are only read from the --heads dict).
conda run --live-stream -n mace_scf python -u scripts/preprocess_data.py \
    --train_file="${MATPES_XYZ}" \
    --valid_fraction="${VALID_FRACTION}" \
    --h5_prefix="${H5_PREFIX}" \
    --r_max="${R_MAX}" \
    --heads="${HEADS}" \
    --compute_statistics \
    --num_process="${NUM_PROCESS}" \
    --seed="${SEED}" \
    --shuffle=True \
    --E0s="{1: -1.11723232, 2: -0.00045595, 3: -0.29734917, 4: -0.04262353, 5: -0.2911712, 6: -1.26281801, 7: -3.12555634, 8: -1.54690765, 9: -0.43794547, 10: -0.01216023, 11: -0.22858276, 12: -0.00994627, 13: -0.21672837, 14: -0.82583191, 15: -1.88719667, 16: -0.89091719, 17: -0.25828681, 18: -0.0235315, 19: -0.17827125, 20: -0.02596217, 21: -2.12966897, 22: -2.40532262, 23: -3.61232779, 24: -5.44620624, 25: -5.14592659, 26: -3.30583367, 27: -1.66614587, 28: -0.28412403, 29: -0.23745594, 30: -0.01098351, 31: -0.19854295, 32: -0.77924665, 33: -1.70136472, 34: -0.78345919, 35: -0.22687512, 36: -0.02265396, 37: -0.16194042, 38: -0.02823145, 39: -2.25679622, 40: -2.23742918, 41: -2.53481909, 42: -4.60213279, 43: -3.40289704, 44: -1.68884293, 45: -1.44016062, 46: -1.47521138, 47: -0.19840574, 48: -0.01374787, 49: -0.19672488, 50: -0.67963499, 51: -1.4302063, 52: -0.6573123, 53: -0.18858477, 54: -0.01020284, 55: -0.13452777, 56: -1.35978407, 57: -0.62794477, 58: -1.43642821, 59: -2.12706895, 60: -3.6389801, 61: -5.08859903, 62: -6.9970228, 63: -9.48606277, 64: -8.11540027, 65: -6.45686224, 66: -5.51640166, 67: -4.30111439, 68: -3.03880565, 69: -2.10513872, 70: -1.84040717, 71: -0.25255978, 72: -3.49292389, 73: -3.5659314, 74: -4.57101127, 75: -4.63436797, 76: -2.88280809, 77: -1.42793567, 78: -0.50244445, 79: -0.18479218, 80: -0.0105212, 81: -0.17939998, 82: -0.63069886, 83: -1.32462383, 89: -0.24210133, 90: -1.04419147, 91: -2.03239022, 92: -4.6443113, 93: -7.30273499, 94: -10.39244586}" \
    --atomic_numbers="[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]" \
