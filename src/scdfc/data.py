from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .config import resolve_path
from .connectivity import config_hash, edges_to_matrix, matrix_to_edges, sliding_window_fc

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Allows preprocessing utilities in a lightweight environment.
    torch = None
    Dataset = object


# ======================== 原始数据发现与审计 ========================
def _subject_from_timeseries(path: Path) -> str:
    """从 `<subject_id>_AAL90_timeseries.csv` 文件名提取被试 ID。"""
    return path.stem.split("_", 1)[0]


def discover_data(config: dict[str, Any]) -> dict[str, Any]:
    """枚举 SC 与各 run 的时间序列，并找出至少有一个 run 的可配对被试。"""
    root = Path(config["paths"]["root"])
    sc_dir = resolve_path(config, "sc_dir")
    sc = {path.stem: path for path in sc_dir.glob("*.csv")}
    runs: dict[str, dict[str, Path]] = {}
    for run, relative in config["paths"]["timeseries"].items():
        directory = Path(relative)
        if not directory.is_absolute():
            directory = root / directory
        runs[run] = {_subject_from_timeseries(path): path for path in directory.glob("*.csv")} if directory.exists() else {}
    all_run_subjects = set().union(*(set(paths) for paths in runs.values()))
    return {"sc": sc, "runs": runs, "subjects": sorted(set(sc).intersection(all_run_subjects))}


def audit_dataset(config: dict[str, Any], sample_limit: int | None = None) -> dict[str, Any]:
    """检查 SC、BOLD、AAL 标签是否能安全进入后续预计算与训练。"""
    found = discover_data(config)
    n_nodes = int(config["data"]["n_nodes"])
    n_timepoints = int(config["data"]["n_timepoints"])
    report: dict[str, Any] = {
        "sc_count": len(found["sc"]),
        "runs": {run: len(paths) for run, paths in found["runs"].items()},
        "paired_subjects_any_run": len(found["subjects"]),
        "run_pair_counts": {run: len(set(paths).intersection(found["sc"])) for run, paths in found["runs"].items()},
        "errors": [],
        "warnings": [],
    }
    atlas_path = resolve_path(config, "atlas_labels")
    atlas_names = [line.split("\t")[1] for line in atlas_path.read_text(encoding="utf-8").splitlines()]
    if len(atlas_names) < n_nodes:
        report["errors"].append(f"Atlas contains {len(atlas_names)} labels, expected at least {n_nodes}")
    atlas_names = atlas_names[:n_nodes]
    sc_zero_fractions: list[float] = []
    for subject in found["subjects"][:sample_limit]:
        sc = pd.read_csv(found["sc"][subject], header=None).to_numpy(dtype=float)
        if sc.shape != (n_nodes, n_nodes):
            report["errors"].append(f"SC {subject} shape is {sc.shape}")
            continue
        if not np.isfinite(sc).all() or np.max(np.abs(sc - sc.T)) > 1e-6:
            report["errors"].append(f"SC {subject} is non-finite or asymmetric")
        sc_zero_fractions.append(float(np.mean(sc == 0)))
        for run, paths in found["runs"].items():
            if subject not in paths:
                continue
            frame = pd.read_csv(paths[subject])
            if frame.shape != (n_timepoints, n_nodes + 1):
                report["errors"].append(f"BOLD {subject}/{run} shape is {frame.shape}")
            if frame.columns[1:].tolist() != atlas_names:
                report["errors"].append(f"ROI order mismatch for {subject}/{run}")
            if not np.isfinite(frame.iloc[:, 1:].to_numpy(dtype=float)).all():
                report["errors"].append(f"Non-finite BOLD values for {subject}/{run}")
    if sc_zero_fractions:
        report["sc_zero_fraction"] = {"median": float(np.median(sc_zero_fractions)), "min": float(np.min(sc_zero_fractions)), "max": float(np.max(sc_zero_fractions))}
    if not found["runs"].get("RL"):
        report["warnings"].append("RL timeseries are not available")
    return report


def write_audit(report: dict[str, Any], path: str | Path) -> None:
    """以 UTF-8 JSON 保存审计结果，供训练前人工确认。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


# ======================== 被试级数据划分 ========================
def make_subject_split(subjects: list[str], fractions: tuple[float, float, float] = (0.70, 0.15, 0.15), seed: int = 20260717) -> pd.DataFrame:
    """按被试随机且可复现地生成 train/val/test 划分。"""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to one")
    subject_ids = np.asarray(sorted(set(map(str, subjects))))
    if len(subject_ids) < 3:
        raise ValueError("At least three unique subjects are required for train/val/test splitting")
    rng = np.random.default_rng(seed)
    rng.shuffle(subject_ids)
    n_train = min(max(int(round(len(subject_ids) * fractions[0])), 1), len(subject_ids) - 2)
    n_val = min(max(int(round(len(subject_ids) * fractions[1])), 1), len(subject_ids) - n_train - 1)
    labels = np.full(len(subject_ids), "test", dtype=object)
    labels[:n_train] = "train"
    labels[n_train : n_train + n_val] = "val"
    return pd.DataFrame({"subject_id": subject_ids, "split": labels}).sort_values("subject_id").reset_index(drop=True)


def validate_split(table: pd.DataFrame) -> None:
    """防止同一 subject_id 重复出现在多个分区。"""
    if table.subject_id.duplicated().any():
        raise ValueError("A subject occurs in more than one split")
    if set(table.split) - {"train", "val", "test"}:
        raise ValueError("Unknown split label")


def load_split(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"subject_id": str})
    validate_split(table)
    return table


# ======================== dFC Zarr 离线缓存 ========================
def _zarr():
    """延迟导入 Zarr，允许仅做数据审计时不安装缓存依赖。"""
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise RuntimeError("Zarr caching requires `pip install zarr<3 numcodecs`") from exc
    return zarr, Blosc


def cache_path(config: dict[str, Any], window_length: int) -> Path:
    return resolve_path(config, "cache_dir") / f"window_{window_length}.zarr"


def precompute_dfc(config: dict[str, Any], window_length: int, subjects: set[str] | None = None, overwrite: bool = False) -> dict[str, int]:
    """将每个 subject/run 的滑窗 Fisher-z FC 写入按窗长分组的 Zarr 缓存。"""
    zarr, Blosc = _zarr()
    found = discover_data(config)
    stride, n_nodes = int(config["data"]["stride"]), int(config["data"]["n_nodes"])
    fisher_clip = float(config["data"]["fisher_clip"])
    destination = cache_path(config, window_length)
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(destination), mode="a")
    settings = {"window_length": window_length, "stride": stride, "n_nodes": n_nodes, "fisher_clip": fisher_clip, "estimator": "rectangular_pearson"}
    settings_hash = config_hash(settings)
    if root.attrs.get("config_hash") not in (None, settings_hash) and not overwrite:
        raise ValueError("Existing cache was generated with different settings; use --overwrite")
    root.attrs.update({**settings, "config_hash": settings_hash})
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    written = skipped = 0
    for run, paths in found["runs"].items():
        for subject, path in sorted(paths.items()):
            if subject not in found["sc"] or (subjects is not None and subject not in subjects):
                continue
            key = f"subjects/{subject}/{run}"
            if key in root and not overwrite:
                skipped += 1
                continue
            values = pd.read_csv(path).iloc[:, 1:].to_numpy(dtype=np.float64)
            fc_z, starts = sliding_window_fc(values, window_length, stride, fisher_clip)
            if key in root:
                del root[key]
            group = root.require_group(key)
            group.create_dataset("fc_z", data=fc_z, chunks=(min(32, len(fc_z)), fc_z.shape[1]), compressor=compressor)
            group.create_dataset("window_starts", data=starts, chunks=(len(starts),), compressor=compressor)
            group.attrs.update({"subject_id": subject, "run": run, "source": str(path.resolve())})
            written += 1
    return {"written": written, "skipped": skipped}


def iter_cached_samples(config: dict[str, Any], window_length: int) -> Iterator[tuple[str, str]]:
    zarr, _ = _zarr()
    root = zarr.open_group(str(cache_path(config, window_length)), mode="r")
    if "subjects" not in root:
        return
    for subject in sorted(root["subjects"].group_keys()):
        for run in sorted(root[f"subjects/{subject}"].group_keys()):
            yield subject, run


def read_cached(config: dict[str, Any], window_length: int, subject: str, run: str) -> tuple[np.ndarray, np.ndarray]:
    zarr, _ = _zarr()
    group = zarr.open_group(str(cache_path(config, window_length)), mode="r")[f"subjects/{subject}/{run}"]
    return np.asarray(group["fc_z"]), np.asarray(group["window_starts"])


# ======================== PyTorch 训练数据集 ========================


def load_sc(config: dict[str, Any], subject: str) -> np.ndarray:
    """读取单名被试原始 SC 矩阵。"""
    path = resolve_path(config, "sc_dir") / f"{subject}.csv"
    return pd.read_csv(path, header=None).to_numpy(dtype=np.float32)


def fit_training_statistics(config: dict[str, Any], window_length: int, output: str | Path) -> dict[str, np.ndarray]:
    """仅使用训练集拟合 SC 标准化参数与未来 FC 群体模板。"""
    split = load_split(resolve_path(config, "split_csv"))
    train_subjects = set(split.loc[split.split == "train", "subject_id"].astype(str))
    sc_edges: list[np.ndarray] = []
    sequences: list[np.ndarray] = []
    seen_sc: set[str] = set()
    for subject, run in iter_cached_samples(config, window_length):
        if subject not in train_subjects:
            continue
        if subject not in seen_sc:
            sc_edges.append(np.log1p(matrix_to_edges(load_sc(config, subject))))
            seen_sc.add(subject)
        fc, _ = read_cached(config, window_length, subject, run)
        sequences.append(fc[1:])
    if not sequences:
        raise ValueError("No cached training sequences found")
    stacked_sc = np.stack(sc_edges)
    stacked_fc = np.stack(sequences)
    stats = {
        "sc_mean": stacked_sc.mean(0).astype(np.float32),
        "sc_std": np.maximum(stacked_sc.std(0), 1e-6).astype(np.float32),
        "fc_mean": stacked_fc.mean((0, 1)).astype(np.float32),
        "fc_std": np.maximum(stacked_fc.std((0, 1)), 1e-6).astype(np.float32),
        "group_template": stacked_fc.mean(0).astype(np.float32),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **stats)
    return stats


class DFCSequenceDataset(Dataset):
    """按 subject/run 返回 SC、首窗 FC 和完整未来 FC 标签。"""
    def __init__(
        self,
        config: dict[str, Any],
        window_length: int,
        split_name: str,
        stats_path: str | Path,
        ablation: str = "full",
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for model datasets")
        self.config = config
        self.window_length = window_length
        self.stats = dict(np.load(stats_path))
        split = load_split(resolve_path(config, "split_csv"))
        allowed = set(split.loc[split.split == split_name, "subject_id"].astype(str))
        self.samples = [(s, r) for s, r in iter_cached_samples(config, window_length) if s in allowed]
        self.ablation = ablation
        self.mean_sc = self.stats["sc_mean"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        subject, run = self.samples[index]
        sc_matrix = load_sc(self.config, subject)
        sc_edges = np.log1p(matrix_to_edges(sc_matrix))
        sc_edges = (sc_edges - self.stats["sc_mean"]) / self.stats["sc_std"]
        if self.ablation == "fc1_only":
            sc_edges = np.zeros_like(sc_edges)
            sc_matrix = np.zeros_like(sc_matrix)
        elif self.ablation == "mean_sc":
            sc_edges = np.zeros_like(sc_edges)
            sc_matrix = edges_to_matrix(np.expm1(self.stats["sc_mean"]), int(self.config["data"]["n_nodes"]), diagonal=0.0)
        elif self.ablation == "shuffled_sc":
            if len(self.samples) < 2:
                raise ValueError("shuffled_sc requires at least two samples")
            candidate = (index + max(1, len(self.samples) // 2)) % len(self.samples)
            while self.samples[candidate][0] == subject:
                candidate = (candidate + 1) % len(self.samples)
            other_subject, _ = self.samples[candidate]
            sc_matrix = load_sc(self.config, other_subject)
            raw = np.log1p(matrix_to_edges(sc_matrix))
            sc_edges = (raw - self.stats["sc_mean"]) / self.stats["sc_std"]
        fc, starts = read_cached(self.config, self.window_length, subject, run)
        warmup = fc[0]
        if self.ablation == "sc_only":
            warmup = np.zeros_like(warmup)
        return {
            "subject_id": subject,
            "run": 0 if run.upper() == "LR" else 1,
            "sc_matrix": torch.from_numpy(sc_matrix.astype(np.float32)),
            "sc_edges": torch.from_numpy(sc_edges.astype(np.float32)),
            "fc_warmup": torch.from_numpy(warmup),
            "fc_future": torch.from_numpy(fc[1:]),
            "window_starts": torch.from_numpy(starts),
        }


class FCWindowDataset(Dataset):
    """从 dFC 序列中可复现地抽取窗口，用于 FC 自编码器训练。"""
    def __init__(self, sequence_dataset: DFCSequenceDataset, windows_per_run: int = 32, seed: int = 0) -> None:
        self.sequence_dataset = sequence_dataset
        self.windows_per_run = windows_per_run
        self.seed = seed

    def __len__(self) -> int:
        return len(self.sequence_dataset) * self.windows_per_run

    def __getitem__(self, index: int):
        run_index = index // self.windows_per_run
        sample = self.sequence_dataset[run_index]
        sequence = torch.cat([sample["fc_warmup"][None], sample["fc_future"]], dim=0)
        generator = np.random.default_rng(self.seed + index)
        window = int(generator.integers(0, len(sequence)))
        return sequence[window]
