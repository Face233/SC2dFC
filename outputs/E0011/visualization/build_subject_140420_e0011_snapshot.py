from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from scdfc.config import load_config
from scdfc.data import DFCSequenceDataset
from scdfc.training import build_sequence_model


ROOT = Path(r"D:\Code\HCP_New")
SUBJECT = "140420"
WINDOW = 83
RUNS = {
    "E0010": ("configs/experiments/E0010_gcn_gru_variance_w3_v1.yaml", "E0010-s42-20260818T075122Z-f073db0"),
    "E0011": ("configs/experiments/E0011_gcn_gru_long_horizon_variance_v1.yaml", "E0011-s42-20260818T083706Z-febe983"),
}
COLORS = {"E0010": "#7570b3", "E0011": "#d95f02"}


def roi_labels(path: Path) -> list[str]:
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields:
            labels.append(fields[1] if len(fields) > 1 else fields[0])
    return labels[:90]


def predict(config_path: str, run_id: str, device: torch.device):
    config = load_config(ROOT / config_path)
    stats = ROOT / "outputs/shared/dataset_lr_v1/window_83/training_stats.npz"
    artifact = ROOT / "outputs/E0003/runs/E0003-s42-20260810T064954Z-4d4337d/checkpoints/best.pt"
    checkpoint = ROOT / f"outputs/{run_id.split('-')[0]}/runs/{run_id}/checkpoints/best.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_sequence_model(
        config, WINDOW, payload["decoder_type"], stats, device,
        payload["sc_encoder_type"], artifact, payload,
    )
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = DFCSequenceDataset(config, WINDOW, "val", stats)
    index = next(i for i, (subject, run) in enumerate(dataset.samples) if subject == SUBJECT and run == "LR")
    sample = dataset[index]
    with torch.no_grad():
        output = model(
            sample["sc_matrix"][None].to(device),
            sample["sc_edges"][None].to(device),
            sample["fc_warmup"][None].to(device),
            steps=len(sample["fc_future"]),
        )
    return config, sample, output.fc_z_edges[0].cpu().numpy()


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = {}
    for name, (config_path, run_id) in RUNS.items():
        config, sample, predictions[name] = predict(config_path, run_id, device)
    target = sample["fc_future"].numpy()
    starts = sample["window_starts"][1:].numpy()
    minutes = (starts - starts[0]) * float(config["data"]["tr_seconds"]) / 60.0
    selected = np.argsort(np.std(target, axis=0))[-6:][::-1]
    labels = roi_labels(ROOT / "data/raw/atlas/ROI_MNI_V4.txt")
    rows, cols = np.triu_indices(90, 1)
    names = [f"{labels[rows[edge]]} – {labels[cols[edge]]}" for edge in selected]

    figure, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    for index, axis in enumerate(axes.ravel()):
        edge = selected[index]
        axis.plot(minutes, target[:, edge], color="#202020", linewidth=1.7, label="True")
        for name in RUNS:
            axis.plot(minutes, predictions[name][:, edge], color=COLORS[name], linewidth=1.3, label=name)
        axis.set_title(names[index])
        axis.set_ylabel("Fisher-z")
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Minutes after first future window")
    handles, legend = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend, loc="upper center", ncol=3)
    figure.suptitle("Subject 140420 — E0011 current best checkpoint", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    image_path = output_dir / "subject_140420_e0010_e0011_current_best.png"
    figure.savefig(image_path, dpi=180)
    plt.close(figure)

    ratios = {
        name: float(np.std(prediction[:, selected]) / np.std(target[:, selected]))
        for name, prediction in predictions.items()
    }
    (output_dir / "subject_140420_e0010_e0011_current_best.json").write_text(
        json.dumps({"subject": SUBJECT, "edge_names": names, "temporal_std_ratio": ratios}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"image": str(image_path), "temporal_std_ratio": ratios}, ensure_ascii=False))


if __name__ == "__main__":
    main()
