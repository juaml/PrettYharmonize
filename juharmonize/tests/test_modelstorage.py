from pathlib import Path
import pytest

from juharmonize.modelstorage import ModelStorage


def test_modelstorage_errors(tmp_path: Path) -> None:
    """Test ModelStorage errors"""
    test_path = tmp_path / "test1"
    test_path.mkdir(exist_ok=True, parents=True)
    (test_path / "test.txt").touch()
    # Test that an error is raised if the path is not empty
    with pytest.raises(ValueError, match="must be empty"):
        ModelStorage(use_disk=True, path=tmp_path)

    # Test that an error is raised if the path is not a directory
    with pytest.raises(ValueError, match="must be a directory"):
        ModelStorage(use_disk=True, path=tmp_path / "test.txt")

    # Test that an error is raised if the path is not writable
    with pytest.raises(ValueError, match="must be writable"):
        ModelStorage(use_disk=True, path="/root")
