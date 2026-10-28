"""Explicit phases for E0052-E0060; defaults to a read-only configuration check.

Train only: python scripts/run_sc_alpha_suite.py --phase train --device cuda
After ALL runs finish: --phase evaluate --device cuda, then --phase summary.
No phase implicitly starts training or external-validation evaluation.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from scdfc.config import load_config
from scdfc.management import config_sha256, validate_experiment_config, verify_data_bindings
from scdfc.sc_alpha import reference_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["check", "train", "evaluate", "summary"], default="check")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    with (ROOT/"reports/experiment_registry.csv").open(encoding="utf-8-sig", newline="") as handle:
        registry = {r["experiment_id"]: r for r in csv.DictReader(handle)}
    configs = {}
    for number in range(52, 61):
        eid = f"E{number:04d}"
        paths = list((ROOT/"configs/experiments").glob(f"{eid}_*.yaml"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one configuration for {eid}")
        config = validate_experiment_config(load_config(paths[0]))
        if registry[eid]["config_sha256"] != config_sha256(config):
            raise RuntimeError(f"{eid} differs from its registered protocol")
        verify_data_bindings(config)
        reference_arrays(config)
        configs[eid] = (paths[0], config)
    print("Nine registered protocols and frozen data/template/alpha bindings verified; no FC fitting performed.", flush=True)
    if args.phase == "check":
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT/"src") + (os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    def command(*arguments):
        subprocess.run([sys.executable, "-m", "scdfc.cli", *arguments], cwd=ROOT, env=env, check=True)
    run_ids = []
    for eid, (path, config) in configs.items():
        completed = []
        for meta_path in (ROOT/"outputs"/eid/"runs").glob("*/metadata.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") == "RUNNING":
                raise RuntimeError(f"{eid} has a RUNNING record; inspect that process before starting another")
            if meta.get("status") == "COMPLETED" and meta.get("seed") == 42:
                if meta["config_sha256"] != config_sha256(config):
                    raise RuntimeError(f"Existing {eid} has another protocol")
                completed.append(meta["run_id"])
        if args.phase == "train":
            if completed:
                print(f"Reusing completed train-only run {sorted(completed)[-1]}", flush=True)
            else:
                arguments = ["run", "--experiment", str(path), "--seed", "42"]
                if args.device: arguments += ["--device", args.device]
                command(*arguments)
        else:
            if not completed:
                raise RuntimeError(f"Complete {eid} training first")
            run_ids.append(sorted(completed)[-1])
    if args.phase == "evaluate":
        from scdfc.sc_alpha_evaluation import require_suite_ready
        require_suite_ready(next(iter(configs.values()))[1])
        for rid in run_ids:
            arguments = ["evaluate-run", "--run-id", rid, "--split", "val"]
            if args.device: arguments += ["--device", args.device]
            command(*arguments)
    if args.phase == "summary":
        command("alpha-summary", "--run-ids", *run_ids)


if __name__ == "__main__":
    main()
