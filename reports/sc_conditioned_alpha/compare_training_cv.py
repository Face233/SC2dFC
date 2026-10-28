"""Compare frozen SC-alpha CV selections with a matching E0032 analytic CV baseline.

Reads training subjects only; does not run external-validation evaluation.
"""
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scdfc.config import load_config
from scdfc.management import file_sha256
from scdfc.models.sc_alpha import sufficient_mse
from scdfc.sc_alpha import fit_templates, load_cohort, statistics
from scdfc.sc_alpha_evaluation import require_suite_ready


def completed_run(eid):
    candidates = []
    for path in (ROOT / "outputs" / eid / "runs").glob("*/metadata.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta["status"] == "COMPLETED" and meta["seed"] == 42:
            candidates.append((meta["run_id"], path.parent))
    return sorted(candidates)[-1][1]


def main():
    config = load_config(next((ROOT / "configs/experiments").glob("E0052_*.yaml")))
    require_suite_ready(config)
    source = completed_run("E0052")
    selection = json.loads((source / "alpha_selection.json").read_text(encoding="utf-8"))
    fold_path = source / "private/cv_folds.json"
    folds = json.loads(fold_path.read_text(encoding="utf-8"))
    if file_sha256(fold_path) != selection["cv_folds_sha256"]:
        raise ValueError("Frozen training-fold checksum differs")
    cohort = load_cohort(config, "train")
    if cohort.subjects != folds["subjects"] or cohort.digest != selection["training_cohort_sha256"]:
        raise ValueError("Training cohort differs from the nine frozen experiments")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = config["evaluation"]["nonoverlap_start"]
    baseline_folds = []
    for i, fold in enumerate(folds["folds"]):
        fitting, heldout = np.asarray(fold["fit"]), np.asarray(fold["heldout"])
        context, template, raw = fit_templates(cohort, fitting)
        alpha = np.clip(raw, 0, 1)
        q, u, v = statistics(cohort, context, template)
        qt, ut, vt = (torch.as_tensor(x, dtype=torch.float32, device=device) for x in (q, u, v))
        a = torch.as_tensor(alpha, dtype=torch.float32, device=device)
        with torch.no_grad():
            mse = float(sufficient_mse(a[None], qt, ut, vt)[heldout, start:].mean())
        values64 = v - 2 * alpha.astype(np.float64)[None] * u + alpha.astype(np.float64)[None] ** 2 * q[:, None]
        baseline_folds.append({"fold": i, "n_heldout": len(heldout), "long_mse": mse,
                               "long_mse_float64": float(values64[heldout, start:].mean()),
                               "clipped_low": int((raw < 0).sum()), "clipped_high": int((raw > 1).sum())})
        print(f"E0032 baseline fold {i+1}/5: {mse:.9f}", flush=True)
    baseline = sum(x["long_mse"] * x["n_heldout"] for x in baseline_folds) / len(cohort.subjects)
    rows = [{"experiment": "E0032", "encoder": "shared analytic alpha", "K": 223,
             "parameters": 0, "lambda": None, "epochs": None, "cv_long_mse": baseline,
             "relative_to_baseline_percent": 0.0}]
    for n in range(52, 61):
        eid = f"E{n:04d}"
        run = completed_run(eid)
        cfg = load_config(next((ROOT / "configs/experiments").glob(f"{eid}_*.yaml")))
        selected = json.loads((run / "alpha_selection.json").read_text(encoding="utf-8"))
        rows.append({"experiment": eid, "encoder": cfg["model"]["sc_encoder"],
                     "K": cfg["model"]["alpha_parameters"],
                     "parameters": selected["initialization"]["parameters"],
                     "lambda": selected["selected_lambda"], "epochs": selected["final_epochs"],
                     "cv_long_mse": selected["cv_long_edge_mse"],
                     "relative_to_baseline_percent": (selected["cv_long_edge_mse"] / baseline - 1) * 100,
                     "selection_path": str((run / "alpha_selection.json").relative_to(ROOT)),
                     "selection_sha256": file_sha256(run / "alpha_selection.json")})
    report = {"scope": "train-only five-fold model-selection diagnostics; not independent validation",
              "subjects": len(cohort.subjects), "training_cohort_sha256": cohort.digest,
              "cv_folds_sha256": selection["cv_folds_sha256"], "device": str(device),
              "nonoverlap_start": start, "baseline_folds": baseline_folds, "rows": rows}
    path = Path(__file__).with_name("training_cv_comparison.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    print(path, flush=True)


if __name__ == "__main__":
    main()
