"""Six frozen-223-alpha / AE residual experiments, with resumable phases.

Default check does not read validation FC or start training. All runs use
managed provenance, seed42, 150 epochs and long validation MSE selection.
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
from scdfc.management import config_sha256, validate_experiment_config
from scdfc.alpha_transfer import verify_transfer_bindings
from scdfc.alpha_transfer_evaluation import summarize_transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["check", "train", "evaluate", "summary", "all"], default="check")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiments", nargs="+", choices=[f"E{n:04d}" for n in range(61, 67)],
                        default=["E0061", "E0062", "E0063"])
    args = parser.parse_args()
    with (ROOT/"reports/experiment_registry.csv").open(encoding="utf-8-sig", newline="") as handle:
        registry = {r["experiment_id"]: r for r in csv.DictReader(handle)}
    configs = {}
    for eid in args.experiments:
        paths = list((ROOT/"configs/experiments").glob(f"{eid}_*.yaml"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one registered config for {eid}")
        config = validate_experiment_config(load_config(paths[0]))
        if registry[eid]["config_sha256"] != config_sha256(config):
            raise ValueError(f"Registered protocol differs: {eid}")
        verify_transfer_bindings(config)
        configs[eid] = (paths[0], config)
    print(f"Verified {len(configs)} AE protocols, frozen 223-alpha components and pinned residual references.", flush=True)
    if args.phase == "check":
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT/"src")
    env["PYTHONUNBUFFERED"] = "1"
    def command(*arguments):
        subprocess.run([sys.executable, "-m", "scdfc.cli", *arguments], cwd=ROOT, env=env, check=True)
    runs = []
    for eid, (path, config) in configs.items():
        complete = []
        for meta_path in (ROOT/"outputs"/eid/"runs").glob("*/metadata.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            same_protocol = meta.get("config_sha256") == config_sha256(config)
            if meta.get("status") == "RUNNING":
                raise RuntimeError(f"Inspect existing RUNNING process before restarting {eid}")
            if not same_protocol and meta.get("status") not in {"FAILED", "ARCHIVED"}:
                raise RuntimeError(f"Existing {eid} uses a different protocol")
            if same_protocol and meta.get("status") == "COMPLETED" and meta.get("seed") == 42:
                complete.append(meta_path.parent)
        if not complete and args.phase in {"train", "all"}:
            print(f"Starting {eid}: frozen {config['artifacts']['sc_alpha']['experiment_id']} + {config['experiment']['baseline']}", flush=True)
            command("run", "--experiment", str(path), "--seed", "42", "--device", args.device)
            complete = [p.parent for p in (ROOT/"outputs"/eid/"runs").glob("*/metadata.json")
                        if json.loads(p.read_text(encoding="utf-8"))["status"] == "COMPLETED"]
        if not complete:
            raise RuntimeError(f"Complete {eid} training first")
        runs.append(sorted(complete)[-1])
        if args.phase in {"train", "all"}:
            protocols = [json.loads((p/"component_protocol.json").read_text(encoding="utf-8")) for p in runs]
            for key in ("initial_parameters_sha256", "initial_cpu_rng_sha256", "initial_cuda_rng_sha256", "sample_manifest_sha256"):
                if len({p[key] for p in protocols}) != 1:
                    raise RuntimeError(f"Residual pairing failed for {key}")
    if args.phase in {"evaluate", "all"}:
        for path in runs:
            report = path/"evaluation_val.json"
            if report.exists():
                print(f"Reusing {report}", flush=True)
            else:
                command("evaluate-run", "--run-id", path.name, "--split", "val", "--device", args.device)
    if args.phase in {"summary", "all"}:
        print(summarize_transfer(ROOT, runs), flush=True)


if __name__ == "__main__":
    main()
