"""Harmonize saved seed-42 validation metrics, without fitting any parameters."""
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from scdfc.config import load_config
from scdfc.data import DFCSequenceDataset
from scdfc.evaluation import _load_model, collect_predictions
from scdfc.history_forecast import aggregate_subjects, forecast_metrics, retrieval_summary
from scdfc.management import file_sha256, verify_artifact, config_sha256

IDS = ["E0036", "E0037", "E0040", "E0041", "E0042", "E0043", "E0045", "E0046", "E0047"]
STATS = ROOT / "outputs/shared/dataset_lr_v1/window_83/training_stats.npz"
OUT = Path(__file__).resolve().parent / "E0036_E0047_validation.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
result = {"split": "val", "seed": 42,
          "definition": "Native forecast horizons; identical history_forecast metric formulas, threshold 1e-6; average within run then subject. Retrieval uses nonoverlap interval, subtracts training template, with additional per-edge time centering for dynamic retrieval.",
          "rows": []}
reference_subjects = None
for eid in IDS:
    reports = list((ROOT / "outputs" / eid / "runs").glob("*/evaluation_val.json"))
    matches = [(p, json.loads((p.parent / "metadata.json").read_text(encoding="utf-8")))
               for p in reports]
    matches = [(p, m) for p, m in matches if m["seed"] == 42 and m["status"] == "COMPLETED"]
    if len(matches) != 1:
        raise ValueError(f"Expected one completed seed42 run for {eid}")
    report_path, metadata = matches[0]
    run_dir = report_path.parent
    config = load_config(run_dir / "config_resolved.yaml")
    if config_sha256(config) != metadata["config_sha256"]:
        raise ValueError(f"Configuration integrity failure for {eid}")
    config["paths"]["root"] = str(ROOT)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = run_dir / "checkpoints/best.pt"
    if eid in {"E0036", "E0037"}:
        print(f"{eid}: validation inference starts", flush=True)
        model, _ = _load_model(config, 83, checkpoint, STATS, device, verify_artifact(config))
        dataset = DFCSequenceDataset(config, 83, "val", STATS, config["experiment"]["ablation"])
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        predictions, targets, subjects, runs = collect_predictions(model, loader, device)
        template = model.group_template.cpu().numpy()
        print(f"{eid}: calculating unified metrics for {len(subjects)} samples", flush=True)
        records = [{"subject_id": s, "run": r, **forecast_metrics(p, t, template, 17)}
                   for p, t, s, r in zip(predictions, targets, subjects, runs)]
        aggregate, per_subject = aggregate_subjects(records)
        for unified, legacy in [("forecast_mse", "mse"), ("forecast_nonoverlap_mse", "long_edge_mse")]:
            if not np.isclose(aggregate[unified], report["aggregate"][legacy], rtol=0, atol=1e-6):
                raise ValueError(f"{eid} inference does not reproduce original {legacy}")
        retrieval = retrieval_summary(predictions, targets, template, 17, subjects)
        subject_ids = sorted(row["subject_id"] for row in per_subject)
        source = "Existing best checkpoint; validation inference with unified metric implementation; native MSE reproduced within 1e-6."
        del model, dataset, loader, predictions, targets, records, per_subject, template
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        if report["checkpoint_sha256"] != file_sha256(checkpoint):
            raise ValueError(f"Checkpoint integrity failure for {eid}")
        aggregate = report["aggregate"]
        retrieval = report["retrieval"]
        subject_ids = sorted(row["subject_id"] for row in report["per_subject"])
        source = "Saved evaluation_val.json; same history_forecast metric implementation."
    if reference_subjects is None:
        reference_subjects = subject_ids
    elif subject_ids != reference_subjects:
        raise ValueError(f"Validation subjects differ in {eid}")
    result["rows"].append({"experiment": eid, "run_id": metadata["run_id"],
                           "checkpoint_sha256": file_sha256(checkpoint),
                           "config_sha256": metadata["config_sha256"],
                           "n_subjects": len(subject_ids),
                           "origin_zero_based": config["data"].get("forecast_origins", [0])[0],
                           "K": config["data"]["warmup_windows"],
                           "H": config["data"].get("forecast_windows", 223),
                           "best_epoch_one_based": report["checkpoint_selection"]["best_epoch"]+1,
                           "aggregate": aggregate, "retrieval": retrieval, "source": source,
                           "original_evaluation": str(report_path)})
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{eid}: done", flush=True)
print(OUT, flush=True)
