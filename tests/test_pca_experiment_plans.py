"""Static PCA plan checks; no training or evaluation is performed."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scdfc.config import load_config
from scdfc.management import create_run_context, file_sha256, validate_experiment_config
from scdfc.managed_cli import command_run


ROOT = Path(__file__).resolve().parents[1]


def plan_path(number):
    return next((ROOT/"configs/experiments").glob(f"E{number:04d}_*.yaml"))


@pytest.mark.parametrize("new,reference", [(50, 36), (51, 37)])
def test_pca_plan_preserves_reference_prediction_protocol(new, reference):
    plan = validate_experiment_config(load_config(plan_path(new)))
    old = validate_experiment_config(load_config(plan_path(reference)))
    for section in ("data", "training", "evaluation", "paths"):
        assert plan[section] == old[section]
    model = dict(plan["model"])
    assert model.pop("fc_representation") == "frozen_pca"
    assert model == old["model"]
    assert plan["experiment"]["ablation"] == old["experiment"]["ablation"]
    assert plan["artifacts"]["alpha_fit"] == old["artifacts"]["alpha_fit"]
    assert plan["artifacts"]["fc_autoencoder"] is None
    pca = plan["artifacts"]["fc_pca"]
    assert file_sha256(ROOT/pca["path"]) == pca["sha256"]
    assert file_sha256(ROOT/pca["manifest_path"]) == pca["manifest_sha256"]
    assert plan["comparison_plan"]["ae_reference"] == f"E{reference:04d}"


def test_pca_pair_only_changes_sc_condition_in_forward_protocol():
    a, b = [load_config(plan_path(n)) for n in (50, 51)]
    for section in ("data", "model", "training", "evaluation", "artifacts"):
        assert a[section] == b[section]
    assert a["experiment"]["ablation"] == "fc1_only"
    assert b["experiment"]["ablation"] == "full"


@pytest.mark.parametrize("number", [50, 51])
def test_design_only_plan_refuses_launch_before_data_or_run_access(number, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("A design-only plan must stop before data/model/run access")
    monkeypatch.setattr("scdfc.managed_cli.verify_data_bindings", unexpected)
    monkeypatch.setattr("scdfc.management.git_metadata", unexpected)
    plan = load_config(plan_path(number))
    plan["experiment"].update(execution_ready=False, execution_blocker="Explicit design-only lock")
    monkeypatch.setattr("scdfc.managed_cli.load_config", lambda *args: deepcopy(plan))
    # Only exercise the refusal branch: no scientific run is started.
    with pytest.raises(RuntimeError, match="Design-only"):
        command_run(SimpleNamespace(experiment=str(plan_path(number)), seed=42, device="cuda"))
    with pytest.raises(RuntimeError, match="Design-only"):
        create_run_context(plan, 42)


def test_pca_configs_are_ready_and_cannot_keep_hidden_ae():
    plan = load_config(plan_path(50))
    assert validate_experiment_config(plan)["experiment"]["execution_ready"] is True
    hidden_ae = deepcopy(plan)
    hidden_ae["artifacts"]["fc_autoencoder"] = {"id": "A0003"}
    with pytest.raises(ValueError, match="without an AE artifact"):
        validate_experiment_config(hidden_ae)
    typo = deepcopy(plan)
    typo["model"]["fc_representation"] = "pca_typo"
    with pytest.raises(ValueError, match="Unknown model.fc_representation"):
        validate_experiment_config(typo)
