"""Audit the observed E0032 residual without fitting on validation data."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from scdfc.config import resolve_path
from scdfc.data import group_template_for_warmup, iter_cached_samples, load_split, read_cached


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
RUN_ID = "E0032-s42-20260923T052241Z-8d7ca92"
RUN_DIR = ROOT / "outputs" / "E0032" / "runs" / RUN_ID
STATS_PATH = ROOT / "outputs" / "shared" / "dataset_lr_v1" / "window_83" / "training_stats.npz"
WINDOW_LENGTH = 83
NONOVERLAP = 17
BOOTSTRAPS = 2000

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 180, "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": False, "grid.alpha": 0.22, "grid.linestyle": "-",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_means(rows: list[dict], seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(rows)
    indices = rng.integers(0, n, size=(BOOTSTRAPS, n))
    values = {key: np.asarray([row[key] for row in rows], dtype=np.float64)
              for key in ("residual_mse", "stable_mse", "dynamic_mse")}
    sampled = {key: value[indices].mean(axis=1) for key, value in values.items()}
    sampled["stable_energy_fraction"] = sampled["stable_mse"] / sampled["residual_mse"]
    result = {}
    for key, value in sampled.items():
        low, high = np.quantile(value, [0.025, 0.975])
        result[key] = [float(low), float(high)]
    return result


def split_half_stability(first: list[np.ndarray], second: list[np.ndarray]) -> dict:
    """Check whether long-horizon mean residuals persist across scan halves."""
    a = np.stack(first).astype(np.float64)
    b = np.stack(second).astype(np.float64)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    similarity = a @ b.T
    n = len(a)
    diagonal = np.diag(similarity)
    other_mean = (similarity.sum(axis=1) - diagonal) / (n - 1)
    ordering = np.argsort(-similarity, axis=1, kind="stable")
    ranks = np.argmax(ordering == np.arange(n)[:, None], axis=1) + 1
    return {
        "early_index": [NONOVERLAP, 120],
        "late_index": [120, 223],
        "within_subject_edge_pearson_mean": float(diagonal.mean()),
        "within_subject_edge_pearson_median": float(np.median(diagonal)),
        "other_subject_edge_pearson_mean": float(other_mean.mean()),
        "same_subject_margin_mean": float(np.mean(diagonal - other_mean)),
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "mean_rank": float(np.mean(ranks)),
    }


class PeriodAccumulator:
    def __init__(self, name: str, start: int, stop: int, edges: int) -> None:
        self.name, self.start, self.stop = name, start, stop
        shape = (stop - start, edges)
        self.sum_r = np.zeros(shape, dtype=np.float64)
        self.sum_r2 = np.zeros(shape, dtype=np.float64)
        self.sum_d = np.zeros(shape, dtype=np.float64)
        self.sum_d2 = np.zeros(shape, dtype=np.float64)
        self.sum_unit_d = np.zeros(shape, dtype=np.float64)
        self.stable_vectors: list[np.ndarray] = []
        self.rows: list[dict] = []

    def add(self, subject: str, run: str, residual: np.ndarray) -> None:
        r = residual[self.start:self.stop]
        m = r.mean(axis=0)
        d = r - m[None, :]
        self.sum_r += r
        self.sum_r2 += r * r
        self.sum_d += d
        self.sum_d2 += d * d
        norm = np.linalg.norm(d)
        if norm > 0:
            self.sum_unit_d += d / norm
        self.stable_vectors.append(m.astype(np.float32))
        left, right = d[:-1].ravel(), d[1:].ravel()
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
        lag1 = float(np.dot(left_centered, right_centered) / denominator) if denominator else float("nan")
        self.rows.append({
            "subject_id": subject, "run": run,
            "residual_mse": float(np.mean(r * r)),
            "stable_mse": float(np.mean(m * m)),
            "dynamic_mse": float(np.mean(d * d)),
            "lag1_pearson": lag1,
            "first_difference_mse": float(np.mean(np.diff(d, axis=0) ** 2)),
        })

    def report(self) -> dict:
        n = len(self.rows)
        if n < 2:
            raise ValueError(f"{self.name} needs at least two subjects")
        mean_r = self.sum_r / n
        mean_r2 = self.sum_r2 / n
        mean_d = self.sum_d / n
        mean_d2 = self.sum_d2 / n
        stable = np.stack(self.stable_vectors).astype(np.float64)
        residual_mse = float(mean_r2.mean())
        stable_mse = float(np.mean([row["stable_mse"] for row in self.rows]))
        dynamic_mse = float(mean_d2.mean())
        pairwise_d = (float(np.sum(self.sum_unit_d ** 2)) - n) / (n * (n - 1))
        identity_error = max(abs(row["residual_mse"] - row["stable_mse"] - row["dynamic_mse"])
                             for row in self.rows)
        return {
            "n_subjects": n, "n_windows": self.stop - self.start,
            "start_index": self.start, "stop_index_exclusive": self.stop,
            "residual_mse": residual_mse,
            "stable_mse": stable_mse,
            "dynamic_mse": dynamic_mse,
            "stable_energy_fraction": stable_mse / residual_mse,
            "dynamic_energy_fraction": dynamic_mse / residual_mse,
            "stable_rms": float(np.sqrt(stable_mse)),
            "dynamic_rms": float(np.sqrt(dynamic_mse)),
            "stable_between_subject_variance": float(np.var(stable, axis=0).mean()),
            "stable_group_bias_mse": float(np.mean(stable.mean(axis=0) ** 2)),
            "dynamic_between_subject_variance": float(np.mean(mean_d2 - mean_d ** 2)),
            "dynamic_shared_trajectory_fraction": float(np.mean(mean_d ** 2) / dynamic_mse),
            "dynamic_pairwise_pearson_mean": pairwise_d,
            "lag1_pearson_mean": float(np.mean([row["lag1_pearson"] for row in self.rows])),
            "first_difference_mse": float(np.mean([row["first_difference_mse"] for row in self.rows])),
            "energy_identity_max_abs_error": float(identity_error),
            "subject_bootstrap_95_ci": bootstrap_means(self.rows),
            "horizon": {
                "index": list(range(self.start, self.stop)),
                "residual_mse": mean_r2.mean(axis=1).tolist(),
                "residual_between_subject_variance": np.maximum(mean_r2 - mean_r ** 2, 0).mean(axis=1).tolist(),
                "dynamic_mse": mean_d2.mean(axis=1).tolist(),
                "dynamic_between_subject_variance": np.maximum(mean_d2 - mean_d ** 2, 0).mean(axis=1).tolist(),
            },
        }


def render_figures(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), sharey=True)
    for axis, period, title in zip(axes, ("full", "long"), ("All future windows", "Non-overlap future")):
        x = np.arange(2)
        for offset, split, color in ((-0.18, "train", "#9a9a9a"), (0.18, "val", "#ff7f0e")):
            record = results[split][period]
            values = [record["stable_mse"], record["dynamic_mse"]]
            intervals = record["subject_bootstrap_95_ci"]
            low = [intervals[key][0] for key in ("stable_mse", "dynamic_mse")]
            high = [intervals[key][1] for key in ("stable_mse", "dynamic_mse")]
            axis.bar(x + offset, values, width=0.34, color=color, label=split.title(),
                     yerr=[np.array(values) - low, np.array(high) - values], capsize=3)
        axis.set_xticks(x, ["Stable", "Dynamic"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Residual MSE contribution")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "E0032_true_residual_components.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for split, color in (("train", "#9a9a9a"), ("val", "#ff7f0e")):
        horizon = results[split]["full"]["horizon"]
        axes[0].plot(horizon["index"], horizon["residual_mse"], color=color, label=split.title())
        axes[1].plot(horizon["index"], horizon["residual_between_subject_variance"], color=color, label=split.title())
    for axis, title in zip(axes, ("Residual MSE", "Between-subject residual variance")):
        axis.axvline(NONOVERLAP, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Future window index")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Mean across FC edges")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "E0032_true_residual_horizon.png", bbox_inches="tight")
    plt.close(fig)


def render_markdown(results: dict) -> None:
    fmt = lambda x: f"{x:.6f}"
    lines = [
        "# E0032 真实残差的稳定与动态成分审计", "",
        "定义：`r[s,t,e]=y[s,t,e]-B[s,t,e]`，其中 `B=g[t,e]+alpha[t]*(FC1[s,e]-g0[e])`。",
        "各区间分别计算 `m[s,e]=mean_t(r)` 与 `d[s,t,e]=r-m`；因此全程与长期的稳定成分定义不同。",
        "所有模板和 alpha 沿用 E0032 训练折产物；验证未来 FC 只用于本次审计。", "",
        "| 划分与区间 | 被试 | 残差 MSE | 稳定 MSE | 动态 MSE | 稳定占比 | 动态被试间相关 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, period in (("train", "full"), ("val", "full"), ("train", "long"), ("val", "long")):
        row = results[split][period]
        lines.append(f"| {split} / {period} | {row['n_subjects']} | {fmt(row['residual_mse'])} | "
                     f"{fmt(row['stable_mse'])} | {fmt(row['dynamic_mse'])} | "
                     f"{100*row['stable_energy_fraction']:.2f}% | {fmt(row['dynamic_pairwise_pearson_mean'])} |")
    val = results["val"]["long"]
    train = results["train"]["long"]
    val_half = results["val"]["stable_split_half"]
    train_half = results["train"]["stable_split_half"]
    ci = val["subject_bootstrap_95_ci"]
    lines.extend([
        "", "## 长期验证集细节", "",
        f"- 长期为未来标签索引 `[17, 223)`，共 {val['n_windows']} 个窗口；稳定成分的 RMS 为 {fmt(val['stable_rms'])}，动态成分的 RMS 为 {fmt(val['dynamic_rms'])}。",
        f"- 稳定占比的被试 bootstrap 95% 区间为 [{100*ci['stable_energy_fraction'][0]:.2f}%, {100*ci['stable_energy_fraction'][1]:.2f}%]。",
        f"- 稳定残差的跨被试方差（对边平均）为 {fmt(val['stable_between_subject_variance'])}；动态残差的跨被试方差（对时距和边平均）为 {fmt(val['dynamic_between_subject_variance'])}。",
        f"- 动态残差的平均跨被试 Pearson 为 {fmt(val['dynamic_pairwise_pearson_mean'])}；共享轨迹能量占比为 {100*val['dynamic_shared_trajectory_fraction']:.2f}%，接近 {val['n_subjects']} 人独立波动的有限样本基准 {100/val['n_subjects']:.2f}%。",
        f"- 相邻未来窗口的动态残差相关为 {fmt(val['lag1_pearson_mean'])}（训练集 {fmt(train['lag1_pearson_mean'])}）。相邻滑窗高度重叠，此值不能单独当作可预测性。",
        f"- 把长期窗口分为 `[17, 120)` 与 `[120, 223)` 两半，真实稳定残差的同被试跨半段边 Pearson 均值为 {fmt(val_half['within_subject_edge_pearson_mean'])}（非本人均值 {fmt(val_half['other_subject_edge_pearson_mean'])}；训练集本人均值 {fmt(train_half['within_subject_edge_pearson_mean'])}）；验证集跨半段身份匹配 Top-1/Top-5 为 {100*val_half['top1']:.2f}% / {100*val_half['top5']:.2f}%，随机期望为 {100/val['n_subjects']:.2f}% / {500/val['n_subjects']:.2f}%。这只检验目标在同一次扫描内的延续性。",
        "- 训练集动态残差的跨被试平均相关略为负值，是训练模板逐时距居中后的有限样本恒等关系，不能解释为反相关现象。",
        "- 稳定 MSE + 动态 MSE = 残差 MSE；这些是使用真实未来 FC 的目标能量，属于理想预测上限的拆分，不代表 FC1 或 SC 已能预测相应成分。",
        "", "## 图", "",
        "![稳定和动态残差能量](E0032_true_residual_components.png)", "",
        "![逐时距残差误差和跨被试方差](E0032_true_residual_horizon.png)", "",
        "完整逐时距数值见 `E0032_true_residual_audit.json`，逐被试指标见 `E0032_true_residual_per_subject.csv`。",
    ])
    (OUT_DIR / "E0032_true_residual_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config_path = RUN_DIR / "config_resolved.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paths"]["root"] = str(ROOT)
    if int(config["data"]["warmup_windows"]) != 1:
        raise ValueError("This audit is defined for one FC warmup window")
    stats = dict(np.load(STATS_PATH))
    template = group_template_for_warmup(stats, 1).astype(np.float64)
    context = stats["context_template"].astype(np.float64)
    alpha_path = RUN_DIR / "alpha_fit.npz"
    alpha = np.load(alpha_path)["alpha"].astype(np.float64)
    original = json.loads((RUN_DIR / "evaluation_val.json").read_text(encoding="utf-8"))
    if not np.allclose(alpha, original["alpha_fit"]["alpha"], rtol=0, atol=1e-12):
        raise ValueError("Saved alpha differs from the E0032 validation report")
    if template.shape != (len(alpha), len(context)):
        raise ValueError("Template, alpha and context dimensions differ")
    split_path = resolve_path(config, "split_csv")
    split_table = load_split(split_path)
    split_of = dict(zip(split_table.subject_id.astype(str), split_table.split))
    periods = {split: {
        "full": PeriodAccumulator("full", 0, len(alpha), len(context)),
        "long": PeriodAccumulator("long", NONOVERLAP, len(alpha), len(context)),
    } for split in ("train", "val")}
    seen = {"train": set(), "val": set()}
    halves = {split: {"early": [], "late": []} for split in periods}
    samples = [(subject, run, split_of.get(subject)) for subject, run in iter_cached_samples(config, WINDOW_LENGTH)]
    samples = [(subject, run, split) for subject, run, split in samples if split in periods]
    for index, (subject, run, split) in enumerate(samples, start=1):
        if subject in seen[split]:
            raise ValueError(f"Multiple runs for {subject}; subject-level aggregation is required")
        seen[split].add(subject)
        fc, _ = read_cached(config, WINDOW_LENGTH, subject, run)
        if fc.shape != (len(alpha) + 1, len(context)):
            raise ValueError(f"Unexpected FC shape for {subject}/{run}: {fc.shape}")
        delta = fc[0].astype(np.float64) - context
        baseline = template + alpha[:, None] * delta[None, :]
        residual = fc[1:].astype(np.float64) - baseline
        for accumulator in periods[split].values():
            accumulator.add(subject, run, residual)
        halves[split]["early"].append(residual[NONOVERLAP:120].mean(axis=0).astype(np.float32))
        halves[split]["late"].append(residual[120:].mean(axis=0).astype(np.float32))
        if index % 50 == 0 or index == len(samples):
            print(f"Residual audit: {index}/{len(samples)}", flush=True)

    if len(seen["train"]) != original["alpha_fit"]["training_samples"] or len(seen["val"]) != original["n_samples"]:
        raise ValueError("Audit cohort does not match the E0032 run")
    results = {split: {period: accumulator.report() for period, accumulator in group.items()}
               for split, group in periods.items()}
    for split, group in halves.items():
        results[split]["stable_split_half"] = split_half_stability(group["early"], group["late"])
    if abs(results["val"]["long"]["residual_mse"] - original["aggregate"]["long_edge_mse"]) > 1e-6:
        raise ValueError("Recomputed validation MSE does not match the E0032 report")
    if abs(results["val"]["full"]["residual_mse"] - original["aggregate"]["mse"]) > 1e-6:
        raise ValueError("Recomputed full validation MSE does not match the E0032 report")

    payload = {
        "schema_version": 1, "experiment": "E0032", "run_id": RUN_ID,
        "definition": "r=y-B; B=g_t+alpha_t*(FC1-g0); m=mean_t(r); d=r-m",
        "fisher_z_edges": len(context), "bootstrap_subject_replicates": BOOTSTRAPS,
        "provenance": {
            "config_resolved_sha256": sha256(config_path),
            "split_csv_sha256": sha256(split_path),
            "training_stats_sha256": sha256(STATS_PATH),
            "alpha_fit_sha256": sha256(alpha_path),
            "dataset_manifest_sha256": config["data"]["manifest_sha256"],
        },
        "splits": results,
    }
    (OUT_DIR / "E0032_true_residual_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT_DIR / "E0032_true_residual_per_subject.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "split", "period", "subject_id", "run", "residual_mse", "stable_mse",
            "dynamic_mse", "lag1_pearson", "first_difference_mse",
        ])
        writer.writeheader()
        for split, group in periods.items():
            for period, accumulator in group.items():
                for row in accumulator.rows:
                    writer.writerow({"split": split, "period": period, **row})
    render_figures(results)
    render_markdown(results)
    print("Validation long:", {key: results["val"]["long"][key] for key in (
        "residual_mse", "stable_mse", "dynamic_mse", "stable_energy_fraction",
        "dynamic_pairwise_pearson_mean")}, flush=True)


if __name__ == "__main__":
    main()
