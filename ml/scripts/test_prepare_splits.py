import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("00_prepare_splits.py")
SPEC = importlib.util.spec_from_file_location("prepare_splits", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
prepare_split = MODULE.prepare_split


def test_split_is_deterministic(tmp_path: Path) -> None:
    category = tmp_path / "pill"
    for relative in ("train/good", "test/good", "test/crack"):
        directory = category / relative
        directory.mkdir(parents=True)
        for index in range(10):
            (directory / f"{index}.png").write_bytes(b"")

    first = prepare_split(tmp_path, "pill", 42)
    second = prepare_split(tmp_path, "pill", 42)

    assert first == second
    assert len(first["train_good"]) == 9
    assert len(first["val_good"]) == 1
    assert set(first["train_good"]).isdisjoint(first["val_good"])

def test_test_split_is_stratified_and_disjoint(tmp_path: Path) -> None:
    category = tmp_path / "pill"
    for relative in ("train/good", "test/good", "test/crack", "test/scratch"):
        directory = category / relative
        directory.mkdir(parents=True)
        for index in range(10):
            (directory / f"{index}.png").write_bytes(b"")

    split = prepare_split(tmp_path, "pill", 42)

    for class_name in ("good", "crack", "scratch"):
        class_split = split["class_splits"][class_name]
        assert len(class_split["calib"]) == 5
        assert len(class_split["eval"]) == 5
        assert set(class_split["calib"]).isdisjoint(class_split["eval"])
    assert set(split["calib"]).isdisjoint(split["eval"])


def test_missing_category_returns_empty_split(tmp_path: Path) -> None:
    split = prepare_split(tmp_path, "missing", 42)
    assert split["train_good"] == []
    assert split["val_good"] == []
    assert split["eval"] == []
