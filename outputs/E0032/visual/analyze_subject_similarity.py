"""Compare cross-subject prediction similarity and identity retrieval on val."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from scdfc.config import resolve_path
from scdfc.data import DFCSequenceDataset, group_template_for_warmup
from scdfc.evaluation import collect_predictions
from scdfc.management import verify_artifact
from scdfc.training import build_sequence_model


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
RUNS = {
    "E0026": "E0026-s42-20260917T041533Z-3e974d5",
    "E0030": "E0030-s42-20260917T122244Z-60c4608",
    "E0031": "E0031-s42-20260918T035339Z-badb7a4",
    "E0032": "E0032-s42-20260923T052241Z-8d7ca92",
}
LABELS = {
    "E0026": "E0026 · dynamic MSE",
    "E0030": "E0030 · retrieval loss",
    "E0031": "E0031 · fixed offset",
    "E0032": "E0032 · fitted decay",
}
NONOVERLAP = 17
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _unit_features(value: torch.Tensor) -> torch.Tensor:
    value = value.reshape(value.shape[0], -1)
    value = value - value.mean(dim=1, keepdim=True)
    return torch.nn.functional.normalize(value, dim=1, eps=1e-12)


def _rank_summary(similarity: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    ordered = np.argsort(-similarity, axis=1, kind="stable")
    ranks = np.empty(similarity.shape[0], dtype=int)
    for index, order in enumerate(ordered):
        ranks[index] = int(np.flatnonzero(order == index)[0]) + 1
    summary = {
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top10": float(np.mean(ranks <= 10)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }
    return ranks.astype(int), summary


def _retrieval(prediction: np.ndarray, target: np.ndarray, group_template: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    pred = prediction[:, NONOVERLAP:].mean(axis=1) - group_template[NONOVERLAP:].mean(axis=0)
    true = target[:, NONOVERLAP:].mean(axis=1) - group_template[NONOVERLAP:].mean(axis=0)
    pred = pred - pred.mean(axis=1, keepdims=True)
    true = true - true.mean(axis=1, keepdims=True)
    pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    true /= np.maximum(np.linalg.norm(true, axis=1, keepdims=True), 1e-12)
    return _rank_summary(pred @ true.T)


def _subject_bootstrap_pair_mean(matrix: np.ndarray, seed: int = 42, n_boot: int = 2000) -> dict[str, float]:
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    point = float(matrix[mask].mean())
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        indices = rng.integers(0, n, size=n)
        sampled = matrix[np.ix_(indices, indices)]
        distinct_subject_pairs = indices[:, None] != indices[None, :]
        if not np.any(distinct_subject_pairs):
            estimates[i] = point
        else:
            estimates[i] = sampled[distinct_subject_pairs].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"mean_off_diagonal": point, "ci_low": float(low), "ci_high": float(high)}


def analyze_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    subjects: list[str],
    group_template: np.ndarray,
    target_dynamic_unit: torch.Tensor | None,
) -> tuple[dict, torch.Tensor]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape or len(subjects) != len(prediction):
        raise ValueError("Prediction, target, and subject dimensions do not agree")

    predicted_dynamic = prediction[:, NONOVERLAP:] - prediction[:, NONOVERLAP:].mean(axis=1, keepdims=True)
    pred_tensor = torch.from_numpy(predicted_dynamic).to(DEVICE)
    predicted_dynamic_unit = _unit_features(pred_tensor)
    if target_dynamic_unit is None:
        true_dynamic = target[:, NONOVERLAP:] - target[:, NONOVERLAP:].mean(axis=1, keepdims=True)
        target_dynamic_unit = _unit_features(torch.from_numpy(true_dynamic).to(DEVICE))
    dynamic_similarity = (predicted_dynamic_unit @ target_dynamic_unit.T).float().cpu().numpy()
    dynamic_ranks, dynamic_topk = _rank_summary(dynamic_similarity)

    prediction_pairwise = (predicted_dynamic_unit @ predicted_dynamic_unit.T).float().cpu().numpy()
    target_pairwise = (target_dynamic_unit @ target_dynamic_unit.T).float().cpu().numpy()
    pairwise_prediction = _subject_bootstrap_pair_mean(prediction_pairwise)
    pairwise_target = _subject_bootstrap_pair_mean(target_pairwise)

    static_prediction_rank, static_topk = _retrieval(prediction, target, group_template)

    other_mask = ~np.eye(len(subjects), dtype=bool)
    other_mean = dynamic_similarity.copy()
    other_mean[~other_mask] = 0.0
    other_mean = other_mean.sum(axis=1) / other_mask.sum(axis=1)
    dynamic_margin = np.diag(dynamic_similarity) - other_mean

    full_mse = float(np.mean((prediction - target) ** 2))
    long_mse_by_subject = np.mean((prediction[:, NONOVERLAP:] - target[:, NONOVERLAP:]) ** 2, axis=(1, 2))
    record = {
        "dynamic_topk": dynamic_topk,
        "static_topk": static_topk,
        "prediction_pairwise_dynamic_pearson": pairwise_prediction,
        "target_pairwise_dynamic_pearson": pairwise_target,
        "mean_diagonal_dynamic_similarity": float(np.diag(dynamic_similarity).mean()),
        "mean_other_subject_dynamic_similarity": float(other_mean.mean()),
        "mean_dynamic_match_margin": float(dynamic_margin.mean()),
        "dynamic_ranks": dynamic_ranks.tolist(),
        "dynamic_match_margin_by_subject": dynamic_margin.tolist(),
        "static_ranks": static_prediction_rank.tolist(),
        "full_mse_recomputed": full_mse,
        "long_mse_by_subject": long_mse_by_subject.tolist(),
    }
    del pred_tensor, predicted_dynamic_unit, dynamic_similarity, prediction_pairwise, target_pairwise
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return record, target_dynamic_unit


def main() -> None:
    stats_path = ROOT / "outputs" / "shared" / "dataset_lr_v1" / "window_83" / "training_stats.npz"
    stats = dict(np.load(stats_path))
    group_template = group_template_for_warmup(stats, 1).astype(np.float32)
    results = {}
    subjects_reference: list[str] | None = None
    target_dynamic_unit = None

    for experiment, run_id in RUNS.items():
        run_dir = ROOT / "outputs" / experiment / "runs" / run_id
        config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
        config["paths"]["root"] = str(ROOT)
        dataset = DFCSequenceDataset(config, 83, "val", stats_path)
        subjects = [dataset.samples[i][0] for i in range(len(dataset))]
        if subjects_reference is None:
            subjects_reference = subjects
        elif subjects != subjects_reference:
            raise ValueError(f"Validation order differs in {experiment}")

        if experiment == "E0032":
            data = [dataset[i] for i in range(len(dataset))]
            warmups = np.stack([x["fc_warmup"].numpy() for x in data]).astype(np.float32)
            targets = np.stack([x["fc_future"].numpy() for x in data]).astype(np.float32)
            del data
            alpha = np.load(run_dir / "alpha_fit.npz")["alpha"].astype(np.float32)
            context = stats.get("context_template", stats["fc_mean"]).astype(np.float32)
            prediction = group_template[None, : targets.shape[1]] + alpha[None, : targets.shape[1], None] * (warmups - context[None])[:, None]
        else:
            checkpoint = run_dir / "checkpoints" / "best.pt"
            artifact_path = verify_artifact(config)
            payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
            model = build_sequence_model(
                config, 83, payload["decoder_type"], stats_path, DEVICE,
                payload.get("sc_encoder_type", "hybrid"), artifact_path, payload,
            )
            incompatible = model.load_state_dict(payload["model"], strict=False)
            if incompatible.unexpected_keys or set(incompatible.missing_keys) - {"context_template"}:
                raise RuntimeError(f"Unexpected checkpoint mismatch: {incompatible}")
            model.eval()
            loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
            prediction, targets, batch_subjects, _ = collect_predictions(model, loader, DEVICE)
            if batch_subjects != subjects:
                raise ValueError(f"Inference order differs in {experiment}")
            del model, loader

        record, target_dynamic_unit = analyze_prediction(
            prediction, targets, subjects, group_template, target_dynamic_unit,
        )
        record["experiment"] = experiment
        record["label"] = LABELS[experiment]
        results[experiment] = record
        print(f"{experiment}: dynamic top1={record['dynamic_topk']['top1']:.3f}, "
              f"top5={record['dynamic_topk']['top5']:.3f}, "
              f"pairwise predicted similarity={record['prediction_pairwise_dynamic_pearson']['mean_off_diagonal']:.3f}, "
              f"recomputed full MSE={record['full_mse_recomputed']:.4f}", flush=True)
        del prediction, targets, dataset
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if subjects_reference is None:
        raise RuntimeError("No validation subjects loaded")
    payload = {
        "schema_version": 1,
        "split": "val",
        "subjects": subjects_reference,
        "n_subjects": len(subjects_reference),
        "nonoverlap_start": NONOVERLAP,
        "dynamic_similarity_definition": "Pearson correlation between each pair of flattened non-overlap trajectories after subtracting each edge's temporal mean per subject",
        "dynamic_topk_definition": "Rank the matching subject's true de-meaned non-overlap dynamic trajectory against true dynamics from all validation subjects using Pearson similarity",
        "static_topk_definition": "Existing future-average FC retrieval; rank each prediction against every validation subject's true future-average FC after subtracting the training group template",
        "chance": {"top1": 1 / len(subjects_reference), "top5": min(5 / len(subjects_reference), 1.0), "top10": min(10 / len(subjects_reference), 1.0)},
        "device": str(DEVICE),
        "methods": results,
    }
    out = OUT_DIR / "subject_similarity_val.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
