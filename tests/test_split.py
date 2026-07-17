import pandas as pd

from scdfc.split import make_family_split, validate_split


def test_family_split_never_separates_relatives():
    subjects = [str(i) for i in range(20)]
    family = pd.DataFrame({"subject_id": subjects, "family_id": [str(i // 2) for i in range(20)]})
    split = make_family_split(subjects, family, seed=3)
    validate_split(split)
    assert split.groupby("family_id").split.nunique().max() == 1
    assert set(split.split) == {"train", "val", "test"}


def test_missing_family_is_rejected():
    family = pd.DataFrame({"subject_id": ["1"], "family_id": ["A"]})
    try:
        make_family_split(["1", "2"], family)
    except ValueError as error:
        assert "Missing family IDs" in str(error)
    else:
        raise AssertionError("Expected missing family validation")

