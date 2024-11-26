import numpy as np
from numpy.testing import assert_array_equal
from pathlib import Path
import pytest

from prettyharmonize.modelstorage import ModelStorage


def test_modelstorage_fs(tmp_path: Path) -> None:
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
        ModelStorage(use_disk=True, path="/")

    test_path = (tmp_path / "test2").as_posix()
    ModelStorage(use_disk=False, path=test_path)


_data = [
    np.array([1, 2, 3]),
    np.array([1, 2, 3]) * 2,
    np.array([1, 2, 3]) * 3,
    np.array([1, 2, 3]) * 4,
    np.array([1, 2, 3]) * 5,
]


def test_modelstorage_memory() -> None:
    """Test ModelStorage in memory"""
    storage = ModelStorage(use_disk=False)

    storage.append(_data[0])
    storage.append(_data[1])
    storage.append(_data[2])
    assert len(storage) == 3
    assert len(storage._mem_models) == 3
    assert_array_equal(storage[0], _data[0])
    assert_array_equal(storage[1], _data[1])
    assert_array_equal(storage[2], _data[2])
    with pytest.raises(IndexError):
        storage[3]
    with pytest.raises(IndexError):
        storage[-4]
    assert_array_equal(storage[-1], _data[2])
    assert_array_equal(storage[-2], _data[1])
    assert_array_equal(storage[-3], _data[0])
    assert_array_equal(storage[0], _data[0])
    assert_array_equal(storage[1], _data[1])
    assert_array_equal(storage[2], _data[2])


def test_modelstorage_disk() -> None:
    """Test ModelStorage in memory"""
    storage = ModelStorage(use_disk=True)

    storage.append(_data[0])
    storage.append(_data[1])
    storage.append(_data[2])
    assert len(storage) == 3
    assert len(storage._mem_models) == 0
    assert_array_equal(storage[0], _data[0])
    assert_array_equal(storage[1], _data[1])
    assert_array_equal(storage[2], _data[2])
    with pytest.raises(IndexError):
        storage[3]
    with pytest.raises(IndexError):
        storage[-4]
    assert_array_equal(storage[-1], _data[2])
    assert_array_equal(storage[-2], _data[1])
    assert_array_equal(storage[-3], _data[0])
    assert_array_equal(storage[0], _data[0])
    assert_array_equal(storage[1], _data[1])
    assert_array_equal(storage[2], _data[2])
