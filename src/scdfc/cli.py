from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, resolve_path
from .data import (
    audit_dataset,
    discover_data,
    fit_training_statistics,
    make_subject_split,
    precompute_dfc,
    validate_split,
    write_audit,
)
from .evaluation import evaluate_checkpoint
from .training import autoencoder_checkpoint_path, train_autoencoder, train_sequence_model


def stats_path(config, window):
    return resolve_path(config, "output_dir") / f"window_{window}" / "training_stats.npz"


def ensure_stats(config, window):
    path = stats_path(config, window)
    if not path.exists():
        fit_training_statistics(config, window, path)
    return path


def command_audit(args):
    config = load_config(args.config)
    report = audit_dataset(config, args.sample_limit)
    output = resolve_path(config, "output_dir") / "audit.json"
    write_audit(report, output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def command_split(args):
    config = load_config(args.config)
    subjects = discover_data(config)["subjects"]
    fractions = (float(config["split"]["train"]), float(config["split"]["val"]), float(config["split"]["test"]))
    split = make_subject_split(subjects, fractions, int(config["seed"]))
    validate_split(split)
    destination = resolve_path(config, "split_csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(destination, index=False)
    print(split.groupby("split").size().to_string())


def command_precompute(args):
    config = load_config(args.config)
    for window in args.windows:
        print(window, precompute_dfc(config, window, overwrite=args.overwrite))


def command_train_ae(args):
    config = load_config(args.config)
    path = train_autoencoder(config, args.window, ensure_stats(config, args.window), args.device)
    print(path)


def command_train(args):
    config = load_config(args.config)
    stats = ensure_stats(config, args.window)
    if not autoencoder_checkpoint_path(config, args.window).exists():
        raise FileNotFoundError("Train the FC autoencoder first with `scdfc train-ae`")
    path = train_sequence_model(
        config,
        args.window,
        args.model,
        stats,
        args.ablation,
        args.device,
        args.sc_encoder,
    )
    print(path)


def command_evaluate(args):
    config = load_config(args.config)
    path = evaluate_checkpoint(config, args.window, args.checkpoint, ensure_stats(config, args.window), args.baseline_checkpoint, args.save_predictions, args.device)
    print(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scdfc")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", default="configs/default.yaml")
    audit.add_argument("--sample-limit", type=int)
    audit.set_defaults(function=command_audit)
    split = subparsers.add_parser("split")
    split.add_argument("--config", default="configs/default.yaml")
    split.set_defaults(function=command_split)
    precompute = subparsers.add_parser("precompute")
    precompute.add_argument("--config", default="configs/default.yaml")
    precompute.add_argument("--windows", nargs="+", type=int, default=[83, 42, 125])
    precompute.add_argument("--overwrite", action="store_true")
    precompute.set_defaults(function=command_precompute)
    train_ae = subparsers.add_parser("train-ae")
    train_ae.add_argument("--config", default="configs/default.yaml")
    train_ae.add_argument("--window", type=int, default=83)
    train_ae.add_argument("--device")
    train_ae.set_defaults(function=command_train_ae)
    train = subparsers.add_parser("train")
    train.add_argument("--config", default="configs/default.yaml")
    train.add_argument("--window", type=int, default=83)
    train.add_argument("--model", choices=["tcn", "transformer", "direct_mlp", "gcn_gru"], required=True)
    train.add_argument("--sc-encoder", choices=["hybrid", "hcp_gcn"])
    train.add_argument("--ablation", choices=["full", "fc1_only", "sc_only", "mean_sc", "shuffled_sc"], default="full")
    train.add_argument("--device")
    train.set_defaults(function=command_train)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", default="configs/default.yaml")
    evaluate.add_argument("--window", type=int, default=83)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--baseline-checkpoint")
    evaluate.add_argument("--save-predictions", action="store_true")
    evaluate.add_argument("--device")
    evaluate.set_defaults(function=command_evaluate)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
