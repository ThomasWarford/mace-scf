"""Write a RESULTS.md for a fit directory: one row per run, read from its logs and metrics.

For each run with validation records (<results>/<run>/<run>_run-*_train.txt) it reports:
- the stages from the run's Configuration log line (name, epochs, config file),
- validation metrics at the lowest-validation-loss epoch of the last stage it validated
  (loss weights change between stages, so losses are only compared within one), in the
  log's "Epoch N:" units (meV/atom, meV/A, meV/A^3 per atom, me, me A, eV),
- the epoch of that stage's final model ("Loading checkpoint: ..._<stage>_epoch-<N>.pt"
  followed by "Loaded model from epoch N" in the log). Training loads its latest
  checkpoint, so with --save_all_checkpoints this is the last epoch, not the best,
- the Slurm jobs that ran it (--name in the first lines of <logs>/*_<jobid>.out).

Metrics are validation metrics as training logged them: SCF stages evaluate from the
reference density, so they are not real-SCF errors.

    python scripts/summarize_fits.py fit_matpes_lsc_r2_full
    python scripts/summarize_fits.py fit_matpes_fixedpoint_vanilla --logs /pscratch/.../logs/fit_matpes_fixedpoint_vanilla

Logs default to <fit_dir>/logs; --logs adds directories (searched in order).
"""

import argparse
import ast
import json
import os
import re
from collections import defaultdict

# (column, key, scale, format): the log's "Epoch N:" units
COLUMNS = [
    ("E MAE", "mae_e_per_atom", 1e3, "{:.1f}"),
    ("E RMSE", "rmse_e_per_atom", 1e3, "{:.1f}"),
    ("F MAE", "mae_f", 1e3, "{:.1f}"),
    ("F RMSE", "rmse_f", 1e3, "{:.1f}"),
    ("S RMSE/atom", "rmse_stress_per_atom", 1e3, "{:.2f}"),
    ("q MAE", "mae_charges", 1e3, "{:.1f}"),
    ("mu MAE", "mae_local_dipoles", 1e3, "{:.1f}"),
    ("Fermi MAE", "mae_fermi_level", 1.0, "{:.3f}"),
]


def read_evals(path):
    """Validation rows keyed by epoch (the file repeats each eval row; keep the last)."""
    evals = {}
    with open(path) as f:
        for line in f:
            if '"mode": "eval"' not in line:
                continue
            row = json.loads(line)
            evals[int(row["epoch"])] = row
    return evals


def find_log(log_dirs, run):
    for d in log_dirs:
        for suffix in ("_run-1.log",):
            path = os.path.join(d, run + suffix)
            if os.path.isfile(path):
                return path
    return None


def _namespace_value(line, key):
    """Value of key=... in an argparse Namespace repr (python literals only)."""
    m = re.search(rf"[ (]{key}=", line)
    if m is None:
        return None
    start = m.end()
    depth, i, quote = 0, start, None
    while i < len(line):
        c = line[i]
        if quote:
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c in "[{(":
            depth += 1
        elif c in "]})":
            if depth == 0:
                break
            depth -= 1
        elif c == "," and depth == 0:
            break
        i += 1
    text = line[start:i]
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def read_log(path):
    """Config path, stages, compute_stress and per-stage selected epochs from a training log."""
    info = {"config": None, "stages": [], "compute_stress": None, "selected": {}}
    pending = None
    with open(path, errors="replace") as f:
        for line in f:
            if "INFO: Configuration: Namespace(" in line and not info["stages"]:
                info["config"] = _namespace_value(line, "config")
                info["compute_stress"] = _namespace_value(line, "compute_stress")
                # stages may hold non-literal objects (FixedPointTrainingOptions(...)), so
                # read name/start/end with a regex rather than literal_eval
                schedule = str(_namespace_value(line, "train_schedule") or "")
                info["stages"] = [
                    (n, int(a), int(b))
                    for n, a, b in re.findall(r"'name': '([^']+)', 'start': (\d+), 'end': (\d+)", schedule)
                ]
                continue
            m = re.search(r"Loading checkpoint: .*_run-\d+_(\S+?)_epoch-(\d+)\.pt", line)
            if m:
                pending = (m.group(1), int(m.group(2)))
                continue
            m = re.search(r"Loaded model from epoch (\d+)", line)
            if m and pending and int(m.group(1)) == pending[1]:
                info["selected"][pending[0]] = pending[1]
                pending = None
    return info


def job_ids(log_dirs):
    """run name -> sorted Slurm job ids, from --name in each job's .out header."""
    jobs = defaultdict(set)
    for d in log_dirs:
        for fname in os.listdir(d):
            m = re.match(r".*_(\d{7,})\.out$", fname)
            if not m:
                continue
            with open(os.path.join(d, fname), errors="replace") as f:
                for _, line in zip(range(60), f):
                    n = re.search(r"\bname='([^']+)'", line)
                    if n:
                        jobs[n.group(1)].add(int(m.group(1)))
                        break
    return {k: sorted(v) for k, v in jobs.items()}


def stage_of(stages, epoch):
    for name, start, end in stages:
        if start <= epoch <= end:
            return name
    return None


def summarize(fit_dir, log_dirs):
    results = os.path.join(fit_dir, "results")
    jobs = job_ids(log_dirs)
    rows = []
    for run in sorted(os.listdir(results)):
        run_dir = os.path.join(results, run)
        if not os.path.isdir(run_dir):
            continue
        train_files = sorted(f for f in os.listdir(run_dir) if f.endswith("_train.txt"))
        evals = {}
        for f in train_files:
            evals.update(read_evals(os.path.join(run_dir, f)))
        if not evals:
            continue
        log = find_log(log_dirs, run)
        info = read_log(log) if log else {"config": None, "stages": [], "compute_stress": None, "selected": {}}
        epochs = sorted(evals)
        # report the last stage that has validation records
        stage = stage_of(info["stages"], epochs[-1])
        in_stage = [e for e in epochs if stage_of(info["stages"], e) == stage] or epochs
        best = min(in_stage, key=lambda e: evals[e]["loss"])
        rows.append({
            "run": run,
            "config": os.path.relpath(info["config"], os.path.abspath(fit_dir)) if info["config"] else None,
            "stages": info["stages"],
            "stage": stage,
            "epochs": (epochs[0], epochs[-1]),
            "best": best,
            "model_epoch": info["selected"].get(stage),
            "metrics": evals[best],
            "compute_stress": info["compute_stress"],
            "jobs": jobs.get(run, []),
            "log": log,
        })
    return rows


def fmt(row, key, scale, form):
    value = row["metrics"].get(key)
    if value is None or (key == "rmse_stress_per_atom" and row["compute_stress"] is False):
        return "–"
    if key == "mae_fermi_level" and value == 0.0:  # direct mode predicts no Fermi level
        return "–"
    return form.format(value * scale)


def write_markdown(fit_dir, rows, out):
    lines = [
        f"# {os.path.basename(os.path.abspath(fit_dir))}: fit results",
        "",
        "Generated by `scripts/summarize_fits.py` from each run's validation records "
        "(`results/<run>/*_train.txt`) and log (`logs/<run>_run-1.log`); rerun it to update.",
        "",
        "- **Best**: lowest-validation-loss epoch of the last stage the run validated; the "
        "metrics are at this epoch. **Model**: epoch of that stage's final `.model` (training "
        "loads its latest checkpoint, the last epoch under `--save_all_checkpoints`); `–`: the "
        "run stopped before writing one.",
        "- **Metrics**: validation set, units as in the training log: E meV/atom, F meV/Å, "
        "S RMSE per atom meV/Å³, q me, mu me Å, Fermi eV. SCF stages are evaluated from the "
        "reference density, not a real SCF. `–`: not computed.",
        "- **Jobs**: Slurm jobs whose `--name` was the run (chained and resumed jobs all listed).",
        "",
        "| Run | Stage | Epochs | Best | Model | " + " | ".join(c[0] for c in COLUMNS) + " | Jobs |",
        "|---|---|---|---|---|" + "---|" * len(COLUMNS) + "---|",
    ]
    for r in rows:
        model_epoch = "–" if r["model_epoch"] is None else str(r["model_epoch"])
        cells = [fmt(r, key, scale, form) for _, key, scale, form in COLUMNS]
        jobs = ", ".join(str(j) for j in r["jobs"]) or "–"
        lines.append(
            f"| `{r['run']}` | {r['stage'] or '–'} | {r['epochs'][0]}–{r['epochs'][1]} | {r['best']} | {model_epoch} | "
            + " | ".join(cells) + f" | {jobs} |"
        )
    lines += ["", "## Stages and configs", ""]
    for r in rows:
        stages = ", ".join(f"{n} {s}–{e}" for n, s, e in r["stages"]) or "–"
        config = f"`{r['config']}`" if r["config"] else "–"
        lines.append(f"- `{r['run']}`: {config}; {stages}")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fit_dir")
    parser.add_argument("--logs", action="append", default=[], help="extra log directory (repeatable)")
    parser.add_argument("-o", "--output", default=None, help="default <fit_dir>/RESULTS.md")
    args = parser.parse_args()
    log_dirs = [d for d in [os.path.join(args.fit_dir, "logs"), *args.logs] if os.path.isdir(d)]
    rows = summarize(args.fit_dir, log_dirs)
    out = args.output or os.path.join(args.fit_dir, "RESULTS.md")
    write_markdown(args.fit_dir, rows, out)
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
