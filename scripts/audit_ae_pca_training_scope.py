"""Reproduce the archived E0003/E0044 scope audit without private FC data.

This checks archived evidence and historical source, not original live caches.
No training, PCA fitting, test FC reading or checkpoint loading is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

import numpy as np
import yaml

from scdfc.data import make_subject_split
from scdfc.management import config_sha256, manifest_digest


OLD_RUN = "outputs/E0003/runs/E0003-s42-20260810T064954Z-4d4337d"
DIAGNOSTIC = "outputs/E0044/diagnostics/E0044-20261020T070318Z-b83fc41d"


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def historical_source(root, path):
    return subprocess.run(["git", "show", f"4d4337d:{path}"], cwd=root, check=True,
                          capture_output=True).stdout.decode("utf-8")


def audit(root):
    root = Path(root).resolve()
    old = root / OLD_RUN
    diagnostic = root / DIAGNOSTIC
    manifest = json.loads((old/"data_manifest.json").read_text(encoding="utf-8"))
    evidence = json.loads((diagnostic/"data_audit.json").read_text(encoding="utf-8"))
    old_config = yaml.safe_load((old/"config_resolved.yaml").read_text(encoding="utf-8"))
    old_metadata = json.loads((old/"metadata.json").read_text(encoding="utf-8"))
    diag_metadata = json.loads((diagnostic/"metadata.json").read_text(encoding="utf-8"))
    training = historical_source(root, "src/scdfc/training.py")
    data = historical_source(root, "src/scdfc/data.py")
    frozen_diagnostic = (diagnostic/"ae_diagnostic.py").read_text(encoding="utf-8")

    sc, lr, records = set(), set(), []
    for record in manifest["files"]:
        name = record["path"].replace("\\", "/")
        if "/sc/" in name:
            sc.add(PurePosixPath(name).stem)
        if "timeseries_lr/" in name:
            subject = PurePosixPath(name).stem.split("_")[0]
            lr.add(subject)
            records.append((subject, record["sha256"]))
    reconstructed = make_subject_split(sorted(sc & lr), seed=42)
    split_hash = digest_bytes(reconstructed.to_csv(index=False, lineterminator="\n").encode())
    split_matches = split_hash == old_config["data"]["split_sha256"] == evidence["split_sha256"]
    if not split_matches:
        raise ValueError("Cannot verify the reconstructed split against the frozen split hash")
    lookup = dict(zip(reconstructed.subject_id, reconstructed.split))
    sets = {fold: set(reconstructed.loc[reconstructed.split == fold, "subject_id"])
            for fold in ("train", "val", "test")}
    sample_checks = {}
    for fold in ("train", "val"):
        samples = evidence["samples"][fold]
        sample_checks[fold] = {
            "count": len(samples), "exact_split_membership": {s for s, _ in samples} == sets[fold],
            "one_lr_run_per_subject": all(r == "LR" for _, r in samples) and len(samples) == len(sets[fold]),
        }
    raw_groups = {}
    for subject, digest in records:
        if subject in lookup:
            raw_groups.setdefault(digest, []).append((subject, lookup[subject]))
    raw_cross = sum(len({fold for _, fold in values}) > 1 for values in raw_groups.values())
    fc_hashes = {fold: {r["sha256"] for r in evidence["cache_content"][fold]} for fold in ("train", "val")}
    log = [json.loads(line) for line in (old/"train.log").read_text(encoding="utf-8").splitlines() if line.strip()]
    last_epoch = max(row["epoch"] for row in log if row.get("event") == "epoch_complete")
    train_count = len(sets["train"])
    unique = [len({int(np.random.default_rng(42+i).integers(0, 224))
                   for i in range(j*32, (j+1)*32)}) for j in range(train_count)]
    archive_checks = {}
    for name in ("data_audit.json", "ae_diagnostic.py", "pca.npz", "summary.json"):
        archive_checks[name] = digest_bytes((diagnostic/name).read_bytes()) == diag_metadata["output_sha256"][name]
    historical_checks = {
        "train_subject_filter": 'split_name, "subject_id"' in data and 'if s in allowed' in data,
        "separate_train_val_datasets": 'window_length, "train", stats_path' in training and 'window_length, "val", stats_path' in training,
        "validation_no_grad": 'with torch.no_grad():' in training and 'model.eval()' in training,
        "train_sampling_32": 'FCWindowDataset(sequence, windows_per_run=32, seed=seed)' in training,
        "val_sampling_8": 'FCWindowDataset(val_sequence, windows_per_run=8, seed=seed)' in training,
        "fixed_index_seed_no_epoch": 'np.random.default_rng(self.seed + index)' in data,
        "edge_loss_was_smooth_l1": 'components["edge"] = F.smooth_l1_loss(prediction, target)' in training,
        "pca_fit_receives_train_samples": '_fit_references(store, samples["train"], config, destination)' in frozen_diagnostic,
        "pca_all_windows": 'shape=(len(samples)*windows, edges)' in frozen_diagnostic and 'pca.fit(matrix)' in frozen_diagnostic,
    }
    required_pass = (all(archive_checks.values()) and all(historical_checks.values())
                     and all(v["exact_split_membership"] and v["one_lr_run_per_subject"] for v in sample_checks.values())
                     and manifest_digest(manifest) == evidence["data_manifest_digest"]
                     and config_sha256(old_config) == old_metadata["config_sha256"])
    return {
        "audit_date": "2026-10-26", "status": "ARCHIVED_EVIDENCE_VERIFIED" if required_pass else "REQUIRES_REVIEW",
        "scope": "historical source + archived manifests/digests; no live private data",
        "historical_commit": old_metadata["git"]["commit"],
        "old_config_hash_matches": config_sha256(old_config) == old_metadata["config_sha256"],
        "manifest_hash_matches": manifest_digest(manifest) == evidence["data_manifest_digest"],
        "reconstructed_split_hash_matches": split_matches,
        "split_counts": dict(Counter(lookup.values())), "diagnostic_samples": sample_checks,
        "train_val_subject_intersection": len(sets["train"] & sets["val"]),
        "cross_split_identical_raw_lr_file_hash_groups": raw_cross,
        "train_val_identical_whole_fc_run_hashes": len(fc_hashes["train"] & fc_hashes["val"]),
        "archived_output_hash_checks": archive_checks, "source_checks": historical_checks,
        "sampling_lr_only_reconstruction": {
            "qualification": "Conditional on original AE cache having exactly these sorted LR runs; old run did not save its actual cache sample manifest",
            "train_slots_per_epoch": train_count*32, "unique_windows": sum(unique),
            "unique_windows_per_subject_range": [min(unique), max(unique)],
            "unique_windows_fixed_across_epochs": True, "full_pca_train_windows": train_count*224,
            "recorded_epochs": last_epoch,
            "old_steps_per_epoch_if_lr_only": int(np.ceil(train_count*32/256)),
            "old_total_steps_if_lr_only": int(np.ceil(train_count*32/256))*last_epoch,
            "new_full_steps_per_epoch": int(np.ceil(train_count*224/256)),
        },
        "loss_audit": {"historical_A0003": "Smooth L1 beta=1, edge weight=1", "current_default": "MSE",
                       "warning": "Old A0003 versus new full-data MSE alone confounds data and loss; use a newly trained sampled-window MSE control"},
        "conclusions": {
            "weight_updates": "Historical code uses train-subject FC only; validation is eval/no_grad for checkpoint selection; no validation/test gradient path found",
            "pca": "Archived PCA path fits all 738 train LR runs only; validation is transform/inverse_transform only",
            "strict_runtime_certification": False,
        },
        "limitations": [
            "Original private split file, raw files, live FC cache, and A0003 weights are absent locally; archived hashes cannot substitute for re-reading those files",
            "Old iter_cached_samples enumerated every cached run without LR whitelist; original AE run did not archive run/window membership, so old LR-only coverage cannot be strictly certified",
            "Whole-run/raw-file exact hash nonduplication does not rule out partial or approximate duplicates",
            "No family identifiers are available; subject-disjoint splitting does not establish family-disjoint splitting",
            "Upstream BOLD processing and historical exploratory test access are outside this audit; do not claim untouched-test or online-causal validation",
            "E0044 ran with a dirty worktree, but relevant computation source and outputs were archived and their declared hashes are checked",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", default="reports/experiment_notes/E0003_E0044_training_scope_audit_20261026.json")
    args = parser.parse_args()
    result = audit(args.root)
    path = Path(args.output)
    if not path.is_absolute():
        path = Path(args.root)/path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
