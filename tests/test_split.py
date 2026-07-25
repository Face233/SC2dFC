from types import SimpleNamespace

import pytest

from scdfc import cli
from scdfc.data import make_subject_split, validate_split


def test_subject_split_is_reproducible_and_complete():
    subjects = [str(i) for i in range(20)]
    split = make_subject_split(subjects, seed=3)
    validate_split(split)
    assert set(split.split) == {"train", "val", "test"}
    assert set(split.subject_id) == set(subjects)
    assert split.equals(make_subject_split(subjects, seed=3))


def test_too_few_subjects_are_rejected():
    try:
        make_subject_split(["1", "2"])
    except ValueError as error:
        assert "At least three" in str(error)
    else:
        raise AssertionError("Expected minimum-subject validation")


def test_legacy_split_refuses_to_overwrite_frozen_file(tmp_path, monkeypatch):
    destination = tmp_path / "split_lr_v1.csv"
    destination.write_text("frozen\n", encoding="utf-8")
    config = {
        "seed": 42,
        "paths": {"root": str(tmp_path), "split_csv": destination.name},
        "split": {"train": 0.7, "val": 0.15, "test": 0.15},
    }
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "discover_data", lambda _: {"subjects": [str(index) for index in range(20)]})

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        cli.command_split(SimpleNamespace(config="unused"))
    assert destination.read_text(encoding="utf-8") == "frozen\n"
