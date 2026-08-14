import pytest

from krsl_ai.data.manifest import Sample, parse_video_name, validate_signer_split


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("P1_gde (1).mov", ("P1", "gde", 1)),
        ("P2_kogda(42).mov", ("P2", "kogda", 42)),
    ],
)
def test_parse_video_name_accepts_dataset_variants(
    filename: str, expected: tuple[str, str, int]
) -> None:
    assert parse_video_name(filename) == expected


def test_parse_video_name_rejects_unknown_name() -> None:
    assert parse_video_name("unexpected.mov") is None


def test_validate_signer_split_rejects_shared_signer() -> None:
    samples = [
        Sample("P1_a_0001", "P1", "a", 1, "P1_a (1).mov", "_a", 1, "train"),
        Sample("P1_b_0001", "P1", "b", 1, "P1_b (1).mov", "_b", 1, "validation"),
        Sample("P2_a_0001", "P2", "a", 1, "P2_a (1).mov", "_a", 1, "test"),
        Sample("P2_b_0001", "P2", "b", 1, "P2_b (1).mov", "_b", 1, "test"),
    ]

    with pytest.raises(ValueError, match="Signers assigned to multiple splits"):
        validate_signer_split(samples)
