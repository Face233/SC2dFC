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
