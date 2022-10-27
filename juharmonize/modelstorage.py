from typing import Optional, Union, Any
import os
from pathlib import Path
import tempfile
import joblib


class ModelStorage:
    """Class to store models in memory or on disk.
    Parameters
    ----------
    use_disk: bool
        If True, store models on disk. If False, store models in memory.
    path: str or pathlib.Path
        Path to store models on disk. If None (default) and use_disk is true,
        a temporary directory will be created. If use_disk is False, this
        parameter is ignored.
    """

    def __init__(
        self, use_disk: bool = False, path: Optional[Union[str, Path]] = None
    ) -> None:
        self.use_disk = use_disk
        if self.use_disk:
            if path is None:
                path = Path(tempfile.mkdtemp())
            else:
                if not isinstance(path, Path):
                    path = Path(path)
                if not path.is_dir():
                    raise ValueError("Path must be a directory")
                if not os.access(path, W_OK):
                    raise ValueError("Path must be writable")
                path.mkdir(exist_ok=True, parents=True)
                if any(path.iterdir()) is True:
                    raise ValueError("Storage path must be empty")
            self.path = path
        self._mem_models = []
        self._n_models = 0

    def append(self, model: Any) -> None:
        """Append a model to the storage.

        Parameters
        ----------
        model: Any
            The model to store.

        """
        if self.use_disk:
            fname = self.path / f"model_{self._n_models}.pkl"
            joblib.dump(model, fname)
        else:
            self._mem_models.append(model)
        self._n_models += 1

    def __len__(self) -> int:
        """Return the number of models stored.

        Returns
        -------
        int
            The number of models stored.
        """
        return self._n_models

    def __getitem__(self, idx: int) -> Any:
        """Return the model at index idx.

        Parameters
        ----------
        idx: int
            The index of the model to return.

        Returns
        -------
        Any
            The model at index idx.
        """
        if idx >= self._n_models:
            raise IndexError("Index out of range")
        if self.use_disk:
            fname = self.path / f"model_{idx}.pkl"
            return joblib.load(fname)
        else:
            return self._mem_models[idx]
